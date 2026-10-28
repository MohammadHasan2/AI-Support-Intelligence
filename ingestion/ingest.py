import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge_base"
ENV_FILE = ROOT / ".env"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250
EMBED_BATCH_SIZE = 16


def load_env():
    if not ENV_FILE.exists():
        raise RuntimeError(f"Missing environment file: {ENV_FILE}")

    for raw_line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url.startswith("https://") or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
        )

    return url, key


def api_request(url, key, path, method="GET", payload=None, params=None,
                prefer=None):
    target = f"{url}{path}"
    if params:
        target += "?" + urlencode(params)

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer

    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(target, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Supabase request failed ({exc.code}) at {path}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Network request failed: {exc.reason}") from exc


def split_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            # Prefer ending on a paragraph or sentence boundary.
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + (2 if text[boundary:boundary + 2] in
                                  ("\n\n", ". ", "! ", "? ") else 0)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = max(start + 1, end - CHUNK_OVERLAP)
        start = next_start

    return chunks


def embed_texts(url, key, texts):
    result = api_request(
        url,
        key,
        "/functions/v1/embed",
        method="POST",
        payload={"texts": texts},
    )

    if result.get("dimensions") != 384:
        raise RuntimeError("Embedding function did not report 384 dimensions")

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RuntimeError("Embedding count does not match input text count")

    for vector in embeddings:
        if not isinstance(vector, list) or len(vector) != 384:
            raise RuntimeError("An embedding has an unexpected shape")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise RuntimeError("An embedding contains non-numeric values")

    return embeddings


def ingest_document(url, key, file_path):
    source_path = file_path.relative_to(KNOWLEDGE_DIR).as_posix()
    content = file_path.read_text(encoding="utf-8-sig").strip()

    if not content:
        print(f"SKIP empty file: {source_path}")
        return 0, 0

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    existing = api_request(
        url,
        key,
        "/rest/v1/documents",
        params={
            "source_path": f"eq.{source_path}",
            "select": "id,content_hash",
        },
    )

    if existing and existing[0]["content_hash"] == content_hash:
        print(f"UNCHANGED {source_path}")
        return 0, 0

    document_rows = api_request(
        url,
        key,
        "/rest/v1/documents",
        method="POST",
        params={"on_conflict": "source_path"},
        payload={
            "source_name": file_path.name,
            "source_path": source_path,
            "content_hash": content_hash,
            "metadata": {"format": file_path.suffix.lstrip(".")},
        },
        prefer="resolution=merge-duplicates,return=representation",
    )

    if not document_rows:
        raise RuntimeError(f"Supabase did not return a document ID: {source_path}")

    document_id = document_rows[0]["id"]
    chunks = split_text(content)

    if not chunks:
        print(f"SKIP no chunks: {source_path}")
        return 0, 0

    # Remove old chunks if this document's content has changed.
    api_request(
        url,
        key,
        "/rest/v1/chunks",
        method="DELETE",
        params={"document_id": f"eq.{document_id}"},
        prefer="return=minimal",
    )

    inserted = 0
    for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[offset:offset + EMBED_BATCH_SIZE]
        vectors = embed_texts(url, key, batch)
        rows = []

        for index, (chunk, vector) in enumerate(
            zip(batch, vectors), start=offset
        ):
            rows.append({
                "document_id": document_id,
                "chunk_index": index,
                "content": chunk,
                "token_count": max(1, len(chunk) // 4),
                "embedding": "[" + ",".join(str(float(v)) for v in vector) + "]",
                "metadata": {
                    "source_name": file_path.name,
                    "source_path": source_path,
                },
            })

        api_request(
            url,
            key,
            "/rest/v1/chunks",
            method="POST",
            params={"on_conflict": "document_id,chunk_index"},
            payload=rows,
            prefer="resolution=merge-duplicates,return=minimal",
        )
        inserted += len(rows)

    print(f"INGESTED {source_path}: {inserted} chunks")
    return 1, inserted


def main():
    url, key = load_env()

    if not KNOWLEDGE_DIR.exists():
        raise RuntimeError(f"Missing folder: {KNOWLEDGE_DIR}")

    files = sorted(
        path for path in KNOWLEDGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )

    if not files:
        raise RuntimeError(f"No .md or .txt files found in {KNOWLEDGE_DIR}")

    changed_docs = 0
    inserted_chunks = 0

    for file_path in files:
        docs, chunks = ingest_document(url, key, file_path)
        changed_docs += docs
        inserted_chunks += chunks

    print("\nIngestion finished.")
    print(f"Files scanned: {len(files)}")
    print(f"Documents added/updated: {changed_docs}")
    print(f"Chunks written: {inserted_chunks}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
