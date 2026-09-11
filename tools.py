import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ddgs import DDGS
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DOCUMENTS_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)


DEFAULT_TOP_K = 6
MAX_WEB_RESULTS = 5
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + 500:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _thread_file(thread_id: str) -> Path:
    safe_id = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    return DOCUMENTS_DIR / f"{safe_id}.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_document(filename: str, data: bytes) -> List[Dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    documents: List[Dict[str, Any]] = []

    if suffix == ".txt":
        text = data.decode("utf-8", errors="ignore")
        for i, chunk in enumerate(_chunk_text(text)):
            documents.append({"text": chunk, "page": None, "chunk": i})
        return documents

    if suffix == ".pdf":
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            page_chunks = _chunk_text(text)
            for chunk_index, chunk in enumerate(page_chunks):
                documents.append({"text": chunk, "page": page_number, "chunk": chunk_index})
        return documents

    raise ValueError("Only .txt and .pdf files are supported.")


def ingest_document(thread_id: str, filename: str, data: bytes) -> Dict[str, Any]:
    document_hash = hashlib.sha256(data).hexdigest()
    existing = _load_json(_thread_file(thread_id), [])

    for doc in existing:
        if doc.get("document_hash") == document_hash:
            return {
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "chunks": len(doc.get("chunks", [])),
                "already_indexed": True,
            }

    chunks = _parse_document(filename, data)
    if not chunks:
        raise ValueError(
            "No extractable text was found. This may be a scanned PDF; OCR is required for image-only PDFs."
        )

    document_id = document_hash[:16]
    record = {
        "document_id": document_id,
        "document_hash": document_hash,
        "filename": filename,
        "chunks": chunks,
    }
    existing.append(record)
    _save_json(_thread_file(thread_id), existing)

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": len(chunks),
        "already_indexed": False,
    }



def list_thread_documents(thread_id: str) -> List[Dict[str, Any]]:
    records = _load_json(_thread_file(thread_id), [])
    return [
        {
            "document_id": record.get("document_id", ""),
            "filename": record.get("filename", "uploaded document"),
            "chunks": len(record.get("chunks", [])),
        }
        for record in records
    ]


def thread_has_documents(thread_id: str) -> bool:
    return bool(list_thread_documents(thread_id))


def _load_thread_chunks(thread_id: str, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    records = _load_json(_thread_file(thread_id), [])
    chunks: List[Dict[str, Any]] = []
    allowed = set(document_ids or [])
    for doc in records:
        if allowed and doc.get("document_id") not in allowed:
            continue
        for item in doc.get("chunks", []):
            chunks.append(
                {
                    "text": item.get("text", ""),
                    "page": item.get("page"),
                    "chunk": item.get("chunk", 0),
                    "filename": doc.get("filename", "uploaded document"),
                    "document_id": doc.get("document_id", ""),
                    "kind": "uploaded",
                }
            )
    return chunks


def _load_local_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in KNOWLEDGE_BASE_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            parsed = _parse_document(path.name, data)
        except (OSError, ValueError):
            continue

        for item in parsed:
            chunks.append(
                {
                    "text": item["text"],
                    "page": item.get("page"),
                    "chunk": item.get("chunk", 0),
                    "filename": str(path.relative_to(KNOWLEDGE_BASE_DIR)),
                    "document_id": "local",
                    "kind": "local",
                }
            )
    return chunks


def _format_source(source_number: int, item: Dict[str, Any], score: float) -> Dict[str, Any]:
    location = item["filename"]
    if item.get("page"):
        location += f", p. {item['page']}"
    return {
        "id": f"S{source_number}",
        "title": location,
        "location": location,
        "url": None,
        "type": item.get("kind", "local"),
        "score": round(float(score), 4),
    }


def search_local_documents(
    thread_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Search local knowledge-base + uploaded documents.

    Uploaded documents are searched even when the query has weak lexical overlap.
    This is important for follow-up questions such as "what does the uploaded PDF say?"
    where TF-IDF can otherwise return zero similarity and incorrectly trigger web fallback.
    """
    uploaded = _load_thread_chunks(thread_id, document_ids=document_ids)
    local = _load_local_chunks()
    corpus = local + uploaded
    corpus = [item for item in corpus if item.get("text", "").strip()]
    if not corpus:
        return []

    texts = [item["text"] for item in corpus]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).ravel()
    except ValueError:
        return []

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    # First keep genuinely relevant matches.
    results: List[Dict[str, Any]] = []
    used_indices = set()
    source_number = 1
    for index, score in ranked:
        if score <= 0:
            continue
        item = corpus[index]
        source = _format_source(source_number, item, float(score))
        results.append({"source": source, "text": item["text"]})
        used_indices.add(index)
        source_number += 1
        if len(results) >= top_k:
            return results

    # Critical fallback: when uploaded documents exist but TF-IDF finds no lexical
    # overlap, return a small amount of uploaded-document evidence instead of
    # immediately abandoning the user's document and searching the open web.
    uploaded_indices = [
        i for i, item in enumerate(corpus)
        if item.get("kind") == "uploaded" and i not in used_indices
    ]
    for index in uploaded_indices[: max(2, min(3, top_k - len(results)))]:
        item = corpus[index]
        source = _format_source(source_number, item, float(scores[index]))
        results.append({"source": source, "text": item["text"]})
        source_number += 1
        if len(results) >= top_k:
            break

    return results


async def web_search(query: str, top_k: int = MAX_WEB_RESULTS) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        def _search():
            with DDGS() as client:
                return client.text(query, max_results=top_k)

        raw_results = await asyncio.to_thread(_search)
        for index, item in enumerate(raw_results or [], start=1):
            title = (item.get("title") or "Untitled source").strip()
            url = (item.get("href") or "").strip()
            body = (item.get("body") or "").strip()
            if not body:
                continue
            results.append(
                {
                    "source": {
                        "id": f"W{index}",
                        "title": title,
                        "location": title,
                        "url": url,
                        "type": "web",
                        "score": None,
                    },
                    "text": body,
                }
            )
    except Exception as exc:
        return [
            {
                "source": {
                    "id": "WERR",
                    "title": "Web search error",
                    "location": "DuckDuckGo",
                    "url": None,
                    "type": "error",
                    "score": None,
                },
                "text": f"Live web search failed: {exc}",
            }
        ]
    return results


def format_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No relevant sources were retrieved."

    blocks = []
    for result in results:
        source = result["source"]
        citation = source["id"]
        title = source["title"]
        url = source.get("url")
        link_line = f"\nURL: {url}" if url else ""
        blocks.append(f"[{citation}] {title}{link_line}\n{result['text']}")
    return "\n\n".join(blocks)
