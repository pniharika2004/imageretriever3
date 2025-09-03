import os
import re
import json
import math
import time
import secrets
import hashlib
from typing import Dict, List, Any, Set

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from .db import (
    init_db,
    insert_document,
    insert_pages_bulk,
    replace_pages_bulk,
    get_documents as db_get_documents,
    get_document as db_get_document,
    search_pages as db_search_pages,
    is_fts_enabled,
    get_pages_for_doc,
    get_page_text,
)
from typing import Optional

# Re-enable vector store (FAISS + FastEmbed)
from .vector_store import (
    init_vector_store,
    index_doc_pages,
    vector_search,
    is_vector_store_ready,
    has_doc as vec_has_doc,
)

try:
    from pypdf import PdfReader  # modern PyPDF2
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install pypdf (pip install pypdf)") from e

try:  # Optional OCR stack
    import fitz  # type: ignore  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore

try:  # Optional OCR
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

try:  # Optional, used by pytesseract image ingestion
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

import io

# Optional: load environment variables from .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

try:  # Local embedding for reranking
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from fastembed import TextEmbedding  # type: ignore
except Exception:  # pragma: no cover
    TextEmbedding = None  # type: ignore

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
PDFS_DIR = os.path.join(PUBLIC_DIR, "pdfs")
os.makedirs(PDFS_DIR, exist_ok=True)

# Load environment from root .env if available (for GROQ_API_KEY, etc.)
if load_dotenv:
    try:
        load_dotenv(os.path.join(BASE_DIR, ".env"))
    except Exception:
        pass


# In-memory doc store
Document = Dict[str, Any]
documents: Dict[str, Document] = {}
# Initialize SQLite database on startup
@app.on_event("startup")
def _startup() -> None:
    init_db()
    init_vector_store(BASE_DIR)
    # Index any existing PDFs in the public folder if not present in DB
    try:
        for name in os.listdir(PDFS_DIR):
            if not name.lower().endswith(".pdf"):
                continue
            doc_id = os.path.splitext(name)[0]
            # Skip if already present
            if db_get_document(doc_id):
                continue
            pdf_path = os.path.join(PDFS_DIR, name)
            extracted = extract_per_page_text(pdf_path)
            pdf_url = f"/pdfs/{name}"
            documents[doc_id] = {
                "pdf_url": pdf_url,
                "pages": extracted["pages"],
                "idf_by_token": extracted["idf_by_token"],
                "num_pages": extracted["num_pages"],
                "filename": name,
            }
            insert_document(doc_id, name, pdf_url, extracted["num_pages"])
            insert_pages_bulk(doc_id, extracted["pages"])
            # Index into vector store
            try:
                index_doc_pages(doc_id, extracted["pages"])
            except Exception:
                pass
        # Ensure vector index for any docs that are in DB but missing in vectors
        try:
            for r in db_get_documents():
                did = r.get("doc_id")
                if did and not vec_has_doc(did):
                    pages = get_pages_for_doc(did)
                    if pages:
                        index_doc_pages(did, pages)
        except Exception:
            pass
    except Exception:
        # Best-effort indexing; continue startup
        pass



STOPWORDS = {
    'the','a','an','and','or','but','if','then','else','of','in','on','at','to','for','from','by','with','as','is','are','was','were','be','been','it','its','that','this','these','those','we','you','they','he','she','his','her','their','our','your','i'
}


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in cleaned.split() if t and t not in STOPWORDS]


def compute_idf(pages_tokens: List[List[str]]) -> Dict[str, float]:
    num_pages = len(pages_tokens)
    doc_freq: Dict[str, int] = {}
    for tokens in pages_tokens:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    idf: Dict[str, float] = {}
    for token, df in doc_freq.items():
        idf[token] = math.log((num_pages + 1) / (df + 1)) + 1
    return idf


def score_pages(pages: List[Dict[str, Any]], idf_by_token: Dict[str, float], query: str) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query)
    q_set = set(q_tokens)
    scores = []
    for page in pages:
        tf: Dict[str, int] = {}
        for tok in page["tokens"]:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in q_set:
            tfv = tf.get(tok, 0)
            if tfv > 0:
                score += (1 + math.log(tfv)) * idf_by_token.get(tok, 1.0)
        scores.append({"page_number": page["page_number"], "score": score})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def build_extractive_summary(pages: List[Dict[str, Any]], page_numbers: List[int], query: str, max_sentences: int = 3) -> str:
    # Collect sentences from the selected pages
    texts: List[str] = []
    for n in page_numbers:
        page = next((p for p in pages if p["page_number"] == n), None)
        if page and page.get("text"):
            texts.append(page["text"]) 
    combined = "\n".join(texts)
    if not combined.strip():
        return ""

    # Split into sentences (very simple splitter)
    sentences = re.split(r"(?<=[\.!?])\s+", combined)
    sentences = [s.strip() for s in sentences if s and len(s.strip()) > 0]
    if not sentences:
        return ""

    # Score sentences by query token overlap (unique tokens)
    q_tokens = set(tokenize(query)) if query else set()
    scored: List[tuple[float, str]] = []
    for s in sentences:
        s_tokens = set(tokenize(s))
        overlap = len(s_tokens & q_tokens) if q_tokens else 0
        # Prefer informative sentences by length (cap length influence)
        length_bonus = min(len(s) / 200.0, 1.0)
        score = overlap * 2.0 + length_bonus
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s for _score, s in scored[:max_sentences]]
    # If no query overlap produced results, fallback to the first few sentences
    if not any(token in tokenize(" ".join(top_sentences)) for token in q_tokens) and q_tokens:
        top_sentences = sentences[:max_sentences]

    summary = " ".join(top_sentences)
    return summary.strip()


def normalize_text(text: str) -> str:
    # Remove common noisy glyphs (private use area) and non-printable characters
    t = re.sub(r"[\uE000-\uF8FF]", " ", text or "")
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_evidence_sentences(pages: List[Dict[str, Any]], query: str, per_page: int = 2, total: int = 6) -> List[Dict[str, Any]]:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return []
    scored_global: List[tuple[float, int, str]] = []  # (score, page_number, sentence)
    for page in pages:
        pn = int(page.get("page_number", 0))
        text = normalize_text(page.get("text", ""))
        if not text:
            continue
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        local_scored: List[tuple[float, str]] = []
        for s in sentences:
            s_norm = s.strip()
            if not s_norm:
                continue
            s_tokens = set(tokenize(s_norm))
            overlap = len(s_tokens & q_tokens)
            if overlap <= 0:
                continue
            length_bonus = min(len(s_norm) / 200.0, 1.0)
            score = overlap * 2.0 + length_bonus
            local_scored.append((score, s_norm))
        local_scored.sort(key=lambda x: x[0], reverse=True)
        for score, s_norm in local_scored[:per_page]:
            scored_global.append((score, pn, s_norm))
    scored_global.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, pn, sent in scored_global[:total]:
        out.append({"page_number": pn, "sentence": sent, "score": float(score)})
    return out


def dense_rerank_pages(candidates: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Optional: Use FastEmbed to re-rank page candidates by dense similarity to the query.
    Falls back to original scores if embedding stack is unavailable.
    """
    if not candidates or np is None or TextEmbedding is None:
        return candidates[:top_k]
    # Build per-page text by concatenating candidate chunks per page
    by_page: Dict[tuple[str, int], str] = {}
    for c in candidates:
        did = str(c.get("doc_id"))
        pn = int(c.get("page_number", 0))
        if pn <= 0:
            continue
        key = (did, pn)
        if key not in by_page:
            by_page[key] = ""
        txt = normalize_text(c.get("text", ""))
        if txt:
            if by_page[key]:
                by_page[key] += "\n"
            by_page[key] += txt
    if not by_page:
        return candidates[:top_k]
    keys = list(by_page.keys())
    corpus = [by_page[k] for k in keys]
    try:
        embedder = TextEmbedding(model_name=os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
        q_emb = None
        for e in embedder.embed([query]):
            if hasattr(e, "tolist"):
                e = e.tolist()
            q_emb = np.array(e, dtype="float32")
            break
        if q_emb is None:
            return candidates[:top_k]
        doc_embs_list: List[List[float]] = []
        for e in embedder.embed(corpus):
            if hasattr(e, "tolist"):
                e = e.tolist()
            doc_embs_list.append(e)
        if not doc_embs_list:
            return candidates[:top_k]
        doc_embs = np.array(doc_embs_list, dtype="float32")
        # cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
        sims = d_norm @ q_norm
        # Rank keys by sims
        order = np.argsort(-sims)
        top_keys = [keys[int(i)] for i in order[:top_k]]
        # Build a filtered list in that order
        ordered: List[Dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for did, pn in top_keys:
            if (did, pn) in seen:
                continue
            seen.add((did, pn))
            # pick any candidate row matching did+pn as the exemplar
            row = next((r for r in candidates if str(r.get("doc_id")) == did and int(r.get("page_number", 0)) == pn), None)
            if row:
                ordered.append(row)
        return ordered
    except Exception:
        return candidates[:top_k]


def _hybrid_candidates(message: str, doc_id: Optional[str]) -> tuple[list[dict[str, Any]], bool, str | None]:
    used_db = False
    backend: str | None = None
    candidates: List[Dict[str, Any]] = []
    # Prefer vector search
    if is_vector_store_ready():
        try:
            candidates = vector_search(message, top_k=8, restrict_doc_id=doc_id) if doc_id else vector_search(message, top_k=8)
            if candidates:
                backend = "vector_faiss"
        except Exception:
            candidates = []
    # Always also try DB FTS for hybrid coverage
    try:
        db_results = db_search_pages(message, limit=8, restrict_doc_id=doc_id) if doc_id else db_search_pages(message, limit=8)
        if db_results:
            used_db = True
            # Merge: keep unique (doc_id, page_number); prefer vector results first
            seen = set((str(r.get("doc_id")), int(r.get("page_number", 0))) for r in candidates)
            for r in db_results:
                key = (str(r.get("doc_id")), int(r.get("page_number", 0)))
                if key not in seen:
                    candidates.append(r)
                    seen.add(key)
            if backend is None:
                backend = "fts5" if is_fts_enabled() else "like"
    except Exception:
        pass
    return candidates[:8], used_db, backend


def extract_per_page_text(pdf_path: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"\s+", " ", text).strip()

        # If no/low text was extracted, try OCR as a best-effort fallback
        if len(text) < 30:
            ocr_text = ocr_pdf_page(pdf_path, idx)
            if ocr_text:
                text = ocr_text

        tokens = tokenize(text)
        pages.append({"page_number": idx, "text": text, "tokens": tokens})
    idf_by_token = compute_idf([p["tokens"] for p in pages])
    return {"pages": pages, "idf_by_token": idf_by_token, "num_pages": len(pages)}


def generate_doc_id() -> str:
    return f"{int(time.time()*1000):x}-{secrets.token_hex(4)}"


# API
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    doc_id = generate_doc_id()
    filename = f"{doc_id}.pdf"
    save_path = os.path.join(PDFS_DIR, filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        extracted = extract_per_page_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    pdf_url = f"/pdfs/{filename}"
    # In-memory cache for fast per-doc scoring (legacy)
    documents[doc_id] = {
        "pdf_url": pdf_url,
        "pages": extracted["pages"],
        "idf_by_token": extracted["idf_by_token"],
        "num_pages": extracted["num_pages"],
        "filename": filename,
    }

    # Persist to DB for cross-document search
    try:
        insert_document(doc_id, filename, pdf_url, extracted["num_pages"])
        insert_pages_bulk(doc_id, extracted["pages"])
        # Index into vector store
        try:
            index_doc_pages(doc_id, extracted["pages"])
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save to DB: {e}")
    return {"docId": doc_id, "pdfUrl": pdf_url, "numPages": extracted["num_pages"]}


@app.get("/api/docs")
async def list_docs():
    try:
        rows = db_get_documents()
        items = [
            {"docId": r["doc_id"], "pdfUrl": r["pdf_url"], "numPages": r["num_pages"]}
            for r in rows
        ]
        return {"documents": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
    message = (payload or {}).get("message")
    doc_id = (payload or {}).get("docId")
    if not message:
        raise HTTPException(status_code=400, detail="Missing message")

    # Gather candidate pages either within a document or across all documents
    candidates: List[Dict[str, Any]] = []  # each: {doc_id, page_number, text}
    doc_meta: Dict[str, Any] | None = None
    used_db = False
    backend: str | None = None
    scope: str = "global" if not doc_id else "single_doc"
    search_backend: str | None = None
    if doc_id:
        # Hybrid (vector + FTS), fallback to in-memory
        candidates, used_db, backend = _hybrid_candidates(message, doc_id)
        search_backend = backend
        # Fallback to in-memory scoring if DB had no results
        doc = documents.get(doc_id)
        if not candidates and doc:
            ranked = score_pages(doc["pages"], doc["idf_by_token"], message)
            top = [r["page_number"] for r in ranked[:3]] or [1]
            for n in top:
                page = next((p for p in doc["pages"] if p["page_number"] == n), None)
                if page:
                    candidates.append({"doc_id": doc_id, "page_number": n, "text": page.get("text", "")})
        # Load doc metadata for image URLs
        doc_meta = db_get_document(doc_id) or ({"doc_id": doc_id, **documents.get(doc_id, {})} if documents.get(doc_id) else None)
        if not doc_meta:
            raise HTTPException(status_code=404, detail="Unknown docId")
    else:
        # Global hybrid
        candidates, used_db, backend = _hybrid_candidates(message, None)
        search_backend = backend
        # If still no candidates, continue and handle with a friendly fallback

    # Build context and image mapping
    doc_to_pages: Dict[str, List[int]] = {}
    context_blocks: List[str] = []
    page_scores_by_doc: Dict[str, Dict[int, float]] = {}
    # Rank candidates: prefer higher vector score, then FTS score, then natural order
    def _cand_score(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("score", 0.0))
        except Exception:
            return 0.0
    candidates_sorted = sorted(candidates, key=_cand_score, reverse=True)

    # Lightweight token overlap filter to drop weak matches
    q_tokens = set(tokenize(message))
    def _overlap(text: str) -> float:
        if not q_tokens:
            return 0.0
        s_tokens = set(tokenize(text or ""))
        inter = len(q_tokens & s_tokens)
        return inter / max(1, len(q_tokens))

    filtered: List[Dict[str, Any]] = []
    for r in candidates_sorted:
        text = r.get("text", "")
        ov = _overlap(text)
        # Dynamic threshold: lower for vector hits, higher for FTS/LIKE
        src = str(r.get("source") or "")
        min_ov = 0.15 if src == "vector" else 0.25
        try:
            sc_ok = float(r.get("score", 0.0)) > 0.0
        except Exception:
            sc_ok = True
        if ov >= min_ov and sc_ok:
            filtered.append(r)
    if filtered:
        candidates_sorted = filtered
    # Dense rerank to get the most semantically similar pages at the top
    try:
        reranked = dense_rerank_pages(candidates_sorted, message, top_k=8)
        if reranked:
            candidates_sorted = reranked
    except Exception:
        pass

    for row in candidates_sorted:
        did = str(row.get("doc_id"))
        pn = int(row.get("page_number", 0))
        if pn <= 0:
            continue
        if did not in doc_to_pages:
            doc_to_pages[did] = []
        if did not in page_scores_by_doc:
            page_scores_by_doc[did] = {}
        if pn not in doc_to_pages[did]:
            doc_to_pages[did].append(pn)
        # Track best score per page
        try:
            sc = float(row.get("score", 0.0))
        except Exception:
            sc = 0.0
        prev = page_scores_by_doc[did].get(pn, float("-inf"))
        if sc > prev:
            page_scores_by_doc[did][pn] = sc
        snippet = normalize_text((row.get("text") or "")[:4000])
        context_blocks.append(f"Doc {did} - Page {pn}:\n{snippet}")
    if not context_blocks:
        # Soft fallback response when nothing indexed or matched
        return JSONResponse({
            "answer": "I couldn't find relevant content yet. Try uploading a PDF or asking a broader question.",
            "needs_image": False,
            "related_pages": [],
            "pdf_url": None,
            "images": [],
            "search_info": {
                "used_db": used_db,
                "backend": backend,
                "scope": scope,
                "candidate_count": 0,
            },
        })
    context = "\n\n".join(context_blocks)

    # Rank pages per document by score (vector/IP or FTS-derived)
    ranked_pages_by_doc: Dict[str, List[int]] = {}
    for did, pages in doc_to_pages.items():
        scores_map = page_scores_by_doc.get(did, {})
        ranked = sorted(pages, key=lambda n: scores_map.get(n, 0.0), reverse=True)
        ranked_pages_by_doc[did] = ranked

    # If querying across all docs, keep only the single most relevant document for images/related pages
    if not doc_id and ranked_pages_by_doc:
        best_did = None
        best_score = float("-inf")
        for did, pages in ranked_pages_by_doc.items():
            if not pages:
                continue
            top_page = pages[0]
            score = page_scores_by_doc.get(did, {}).get(top_page, 0.0)
            if score > best_score:
                best_score = score
                best_did = did
        if best_did is not None:
            ranked_pages_by_doc = {best_did: ranked_pages_by_doc.get(best_did, [])}

    answer = ""
    needs_image = False

    # For backward compatibility when a single doc is targeted
    first_doc_id = next(iter(ranked_pages_by_doc.keys())) if ranked_pages_by_doc else (doc_id or "")
    related_pages: List[int] = ranked_pages_by_doc.get(first_doc_id, [])[:3] if first_doc_id else []

    # Try Groq LLM to produce structured JSON answer, fallback to extractive
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided document context. "
        "If an image/figure/diagram/table is directly relevant, set needs_image to true and include related page numbers. "
        "Respond in STRICT JSON only with keys: answer (string), needs_image (boolean), related_pages (array of integers). "
        "Do not include any extra commentary."
    )
    # Build additional evidence from top candidate pages to ground the LLM
    pages_for_summary = [
        {"page_number": int(r.get("page_number", 0)), "text": r.get("text", "")}
        for r in candidates_sorted[:8]
    ]
    evidence = build_evidence_sentences(pages_for_summary, message, per_page=2, total=6)
    evidence_block = "\n".join([f"Page {e['page_number']}: {e['sentence']}" for e in evidence])

    user_prompt = (
        "Answer the user's question using only the provided context and evidence. "
        "Return STRICT JSON: {\"answer\": string, \"needs_image\": boolean, \"related_pages\": number[]}\n\n"
        f"Question: {message}\n\nDocument context:\n{context}\n\nEvidence:\n{evidence_block}"
    )

    answer = ""
    needs_image = False

    try:
        if Groq is None:
            raise RuntimeError("groq sdk not installed")
        # GROQ_API_KEY can be provided via env var; client auto-reads
        client = Groq()
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama3-70b-8192"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = ""
        try:
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
        except Exception:
            content = ""
        json_text = content
        m = re.search(r"\{[\s\S]*\}", json_text)
        if m:
            json_text = m.group(0)
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("answer"), str):
                answer = parsed["answer"]
            if isinstance(parsed.get("needs_image"), bool):
                needs_image = parsed["needs_image"]
            # related_pages from model only applies within the first doc for compatibility
            if isinstance(parsed.get("related_pages"), list) and parsed["related_pages"] and first_doc_id:
                rp = []
                if doc_meta is None and first_doc_id:
                    doc_meta = db_get_document(first_doc_id)
                max_pages = (doc_meta or {}).get("num_pages", 0)
                for n in parsed["related_pages"]:
                    try:
                        num = int(n)
                        if 1 <= num <= (max_pages or 10_000):
                            rp.append(num)
                    except Exception:
                        continue
                if rp:
                    related_pages = rp[:3]
    except Exception:
        # Non-LLM extractive answer over the candidate pages
        pages_for_summary = [
            {"page_number": int(r.get("page_number", 0)), "text": r.get("text", "")}
            for r in candidates
        ]
        summary = build_extractive_summary(
            pages_for_summary,
            [p["page_number"] for p in pages_for_summary if p.get("page_number")][:3],
            message,
            max_sentences=3,
        )
        answer = summary or "No clear answer found in the provided pages."
        # Heuristic for images
        q_lower = str(message).lower()
        needs_image = bool(related_pages) or any(w in q_lower for w in ["image", "figure", "table", "diagram", "chart", "graph"])  # noqa: E501

    # Ensure we always provide a non-empty answer
    if not (isinstance(answer, str) and answer.strip()):
        pages_for_summary = [
            {"page_number": int(r.get("page_number", 0)), "text": r.get("text", "")}
            for r in candidates
        ]
        summary = build_extractive_summary(pages_for_summary, [p["page_number"] for p in pages_for_summary if p.get("page_number")][:3], message, max_sentences=3)
        answer = summary or "No clear answer found in the provided pages."
        q_lower = str(message).lower()
        needs_image = bool(related_pages) or any(w in q_lower for w in ["image", "figure", "table", "diagram", "chart", "graph"])  # noqa: E501

    # Build images payload with all related pages per document (deduped above)
    images: List[Dict[str, Any]] = []
    seen_signatures: Set[str] = set()
    for did, pages in ranked_pages_by_doc.items():
        meta = db_get_document(did)
        # If missing in DB, try in-memory cache
        if not meta and documents.get(did):
            meta = {"doc_id": did, "pdf_url": documents[did]["pdf_url"]}
        if not meta:
            continue
        deduped_pages: List[int] = []
        for pn in pages:
            text = get_page_text(did, pn)
            if text is None and documents.get(did):
                # fallback to in-memory cache
                page = next((p for p in documents[did]["pages"] if p["page_number"] == pn), None)
                text = (page or {}).get("text", "")
            norm = re.sub(r"\s+", " ", (text or ""))[:5000].strip().lower()
            sig = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            deduped_pages.append(pn)
        if deduped_pages:
            images.append({"docId": did, "pdf_url": meta.get("pdf_url"), "pages": deduped_pages[:3]})

    # Backward compatibility for existing frontend
    pdf_url_out = None
    if doc_id and doc_meta:
        pdf_url_out = doc_meta.get("pdf_url")
    elif images:
        pdf_url_out = images[0].get("pdf_url")

    return JSONResponse({
        "answer": answer,
        "needs_image": needs_image,
        "related_pages": related_pages,
        "pdf_url": pdf_url_out,
        "images": images,
        "search_info": {
            "used_db": used_db,
            "backend": search_backend or backend,
            "scope": scope,
            "candidate_count": len(candidates),
        },
    })


@app.post("/api/reindex")
async def reindex_all() -> Dict[str, Any]:
    # Re-extract text (with OCR) and rebuild FAISS vectors for all docs
    updated: List[str] = []
    try:
        rows = db_get_documents()
        for r in rows:
            did = r.get("doc_id")
            filename = r.get("filename") or r.get("pdf_url", "").split("/")[-1]
            if not did or not filename:
                continue
            pdf_path = os.path.join(PDFS_DIR, filename)
            if not os.path.exists(pdf_path):
                continue
            extracted = extract_per_page_text(pdf_path)
            replace_pages_bulk(did, extracted["pages"])
            try:
                index_doc_pages(did, extracted["pages"])  # type: ignore[arg-type]
            except Exception:
                pass
            updated.append(did)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")
    return {"updated": updated, "count": len(updated)}


def ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """Best-effort OCR for a single page using PyMuPDF render + Tesseract.

    Returns empty string if OCR stack is unavailable or any failure happens.
    """
    if fitz is None or pytesseract is None or Image is None:
        return ""
    try:
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(max(0, page_number - 1))
            # Render at higher DPI for better OCR
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
        finally:
            doc.close()
        img = Image.open(io.BytesIO(img_bytes))  # type: ignore[arg-type]
        text = pytesseract.image_to_string(img)
        return re.sub(r"\s+", " ", text or "").strip()
    except Exception:
        return ""


# Static files
# Mount API first, then static at '/'
app.mount("/pdfs", StaticFiles(directory=PDFS_DIR), name="pdfs")
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")


