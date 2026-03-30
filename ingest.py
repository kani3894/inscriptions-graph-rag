#!/usr/bin/env python3
"""
South Indian Inscriptions Graph RAG — Ingestion Pipeline

Scans the inscriptions archive in Obsidian, extracts historical entities
(dynasties, kings, temples, locations, dates) using ollama qwen3:1.7b, builds a
NetworkX graph, and embeds chunks with all-MiniLM-L6-v2.

Incremental: only processes new/modified files.
"""

import json
import gc
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests as http_requests
import networkx as nx
import numpy as np

CONCURRENCY = 20  # parallel OpenRouter requests

# --- Config ---
VAULT_DIR = Path(os.path.expanduser("~/Obsidian/Kani/South-Indian-Inscriptions"))
DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "state.json"
GRAPH_FILE = DATA_DIR / "graph.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"

# Skip patterns
SKIP_PATTERNS = [
    ".obsidian",
    "_site-map.md",
]

# Chunk settings
MAX_CHUNK_TOKENS = 600  # ~600 tokens ≈ 2400 chars
CHUNK_OVERLAP = 200     # chars overlap between chunks

BATCH_SIZE = 128  # embedding batch size (MiniLM is small, can handle large batches)

# OpenRouter config
LLM_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
LLM_MODEL = "meta-llama/llama-3.3-70b-instruct"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

# Entity extraction prompt tuned for historical inscriptions
EXTRACTION_PROMPT = """Extract entities and relationships from this South Indian inscription. Return JSON only.
{"entities":[{"name":"...","type":"..."}],"relationships":[{"source":"...","target":"...","type":"..."}]}
Types: person, dynasty, temple, deity, location, admin_division, date, land_grant, title
Rel types: ruled, built, donated, son_of, succeeded, located_in, part_of, feudatory_of, granted, belongs_to
If nothing found return {"entities":[],"relationships":[]}"""


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"files": {}, "last_run": None}


def save_state(state):
    state["last_run"] = datetime.now().isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_graph():
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE) as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    return nx.DiGraph()


def save_graph(G):
    data = nx.node_link_data(G)
    with open(GRAPH_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_chunks():
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE) as f:
            return json.load(f)
    return {"chunks": []}


def save_chunks(chunks_data):
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks_data, f)


def load_embeddings():
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE) as f:
            return json.load(f)
    return {"model": "all-MiniLM-L6-v2", "dimensions": 384, "total": 0, "embeddings": []}


def save_embeddings(emb_data):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(emb_data, f)


def find_files():
    """Find all markdown files in the inscriptions archive."""
    files = []
    for f in VAULT_DIR.rglob("*.md"):
        rel = str(f.relative_to(VAULT_DIR))
        if any(skip in rel for skip in SKIP_PATTERNS):
            continue
        files.append(f)
    return files


def chunk_text(text, source_path):
    """Split text into chunks respecting section boundaries."""
    chunks = []
    sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)

    current_chunk = ""
    chunk_idx = 0

    for section in sections:
        if not section.strip():
            continue
        if len(current_chunk) + len(section) > MAX_CHUNK_TOKENS * 4:
            if current_chunk.strip():
                chunk_id = hashlib.md5(f"{source_path}:{chunk_idx}".encode()).hexdigest()[:12]
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "source": str(source_path),
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1
            current_chunk = section
        else:
            current_chunk += section

    if current_chunk.strip():
        chunk_id = hashlib.md5(f"{source_path}:{chunk_idx}".encode()).hexdigest()[:12]
        chunks.append({
            "id": chunk_id,
            "text": current_chunk.strip(),
            "source": str(source_path),
            "chunk_idx": chunk_idx,
        })

    if not chunks and text.strip():
        for i in range(0, len(text), MAX_CHUNK_TOKENS * 4 - CHUNK_OVERLAP):
            segment = text[i:i + MAX_CHUNK_TOKENS * 4]
            if segment.strip():
                chunk_id = hashlib.md5(f"{source_path}:{chunk_idx}".encode()).hexdigest()[:12]
                chunks.append({
                    "id": chunk_id,
                    "text": segment.strip(),
                    "source": str(source_path),
                    "chunk_idx": chunk_idx,
                })
                chunk_idx += 1

    return chunks


def extract_frontmatter(text):
    meta = {}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            fm = text[3:end].strip()
            for line in fm.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta


def extract_entities(text, source_path):
    """Use Groq (llama-3.3-70b) to extract entities and relationships."""
    if len(text) > 3000:
        text = text[:3000]

    user_msg = f"Source: {source_path}\n\n{text}"

    for attempt in range(4):
        try:
            resp = http_requests.post(
                LLM_URL,
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 0))
                wait = max(retry_after, 60 * (attempt + 1))
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/4)...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]

            return json.loads(content)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Warning: entity extraction failed for {source_path}: {e}")
            return {"entities": [], "relationships": []}
        except http_requests.RequestException as e:
            body = ""
            if hasattr(e, 'response') and e.response is not None:
                body = e.response.text[:200]
            print(f"  Warning: LLM request failed: {e} {body}")
            return {"entities": [], "relationships": []}
    print(f"  Warning: exhausted retries for {source_path}")
    return {"entities": [], "relationships": []}


def add_to_graph(G, entities_data, source_path):
    source_node = str(source_path)
    G.add_node(source_node, type="document", name=Path(source_path).stem)

    for entity in entities_data.get("entities", []):
        name = entity.get("name", "").strip()
        if not name:
            continue
        node_id = name.lower().replace(" ", "_")

        if G.has_node(node_id):
            G.nodes[node_id].setdefault("sources", [])
            if source_node not in G.nodes[node_id]["sources"]:
                G.nodes[node_id]["sources"].append(source_node)
            G.nodes[node_id]["mentions"] = G.nodes[node_id].get("mentions", 0) + 1
        else:
            G.add_node(node_id,
                       name=name,
                       type=entity.get("type", "unknown"),
                       description=entity.get("description", ""),
                       sources=[source_node],
                       mentions=1)

        G.add_edge(node_id, source_node, type="mentioned_in")

    for rel in entities_data.get("relationships", []):
        src = rel.get("source", "").strip().lower().replace(" ", "_")
        tgt = rel.get("target", "").strip().lower().replace(" ", "_")
        if src and tgt and G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt,
                       type=rel.get("type", "related_to"),
                       context=rel.get("context", ""),
                       source=source_node)


def main():
    print("=" * 60)
    print("South Indian Inscriptions — Graph RAG Ingestion")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    G = load_graph()
    chunks_data = load_chunks()
    emb_data = load_embeddings()

    all_files = find_files()
    print(f"Found {len(all_files)} files in archive")

    new_files = []
    for f in all_files:
        rel_path = str(f.relative_to(VAULT_DIR))
        mtime = f.stat().st_mtime
        prev_mtime = state["files"].get(rel_path, {}).get("mtime", 0)
        if mtime > prev_mtime:
            new_files.append((f, rel_path))

    print(f"New/modified files: {len(new_files)}")

    if not new_files:
        print("Nothing to process. Done.")
        return

    skip_entities = "--skip-entities" in sys.argv
    trial_limit = None
    for arg in sys.argv:
        if arg.startswith("--trial="):
            trial_limit = int(arg.split("=")[1])
    if skip_entities:
        print("Skipping entity extraction (--skip-entities flag)")
    elif not LLM_API_KEY:
        print("WARNING: No OPENROUTER_API_KEY set. Skipping entity extraction.")
        skip_entities = True
    else:
        print(f"Using OpenRouter ({LLM_MODEL}) for entity extraction")
        if trial_limit:
            print(f"  Trial mode: processing max {trial_limit} files")
            new_files = new_files[:trial_limit]

    new_chunks = []
    processed = 0

    # Pre-read all files and chunk them (fast, local I/O)
    file_tasks = []  # (filepath, rel_path, text, file_chunks)
    for filepath, rel_path in new_files:
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  Skip (read error): {e}")
            continue

        if not text.strip() or len(text.strip()) < 50:
            state["files"][rel_path] = {"mtime": filepath.stat().st_mtime, "chunks": 0}
            continue

        meta = extract_frontmatter(text)
        chunks_data["chunks"] = [c for c in chunks_data["chunks"] if c.get("source") != rel_path]

        file_chunks = chunk_text(text, rel_path)
        for c in file_chunks:
            c["title"] = meta.get("title", Path(rel_path).stem)
            c["date"] = meta.get("crawled", "")

        chunks_data["chunks"].extend(file_chunks)
        new_chunks.extend(file_chunks)
        file_tasks.append((filepath, rel_path, text, file_chunks))

    print(f"Files to extract entities from: {len(file_tasks)}")

    # Parallel entity extraction
    def process_one(idx, filepath, rel_path, text):
        entities = extract_entities(text, rel_path)
        return idx, filepath, rel_path, entities

    if not skip_entities:
        batch_start = 0
        while batch_start < len(file_tasks):
            batch = file_tasks[batch_start:batch_start + CONCURRENCY]
            futures = {}
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                for i, (filepath, rel_path, text, file_chunks) in enumerate(batch):
                    idx = batch_start + i
                    futures[executor.submit(process_one, idx, filepath, rel_path, text)] = (filepath, rel_path, file_chunks)

                for future in as_completed(futures):
                    idx, filepath, rel_path, entities = future.result()
                    n_ent = len(entities.get("entities", []))
                    n_rel = len(entities.get("relationships", []))
                    add_to_graph(G, entities, rel_path)
                    processed += 1

                    state["files"][rel_path] = {
                        "mtime": filepath.stat().st_mtime,
                        "chunks": len(futures[future][2]),
                        "entities": n_ent,
                    }

                    if processed <= 5 or processed % 100 == 0:
                        print(f"  [{processed}/{len(file_tasks)}] {rel_path} — {n_ent} entities, {n_rel} rels")

            batch_start += CONCURRENCY

            if processed % 50 < CONCURRENCY:
                print(f"  [Checkpoint: {processed}/{len(file_tasks)} files]")
                save_state(state)
                save_graph(G)
                save_chunks(chunks_data)
    else:
        for filepath, rel_path, text, file_chunks in file_tasks:
            state["files"][rel_path] = {
                "mtime": filepath.stat().st_mtime,
                "chunks": len(file_chunks),
                "entities": 0,
            }

    # Embed new chunks
    if new_chunks:
        print(f"\n--- Embedding {len(new_chunks)} new chunks ---")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        emb_data["embeddings"] = [e for e in emb_data["embeddings"]
                                   if e["id"] not in {c["id"] for c in new_chunks}]

        all_texts = [c["text"] for c in new_chunks]
        print(f"  Encoding all {len(all_texts)} chunks in one shot...")
        all_embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=BATCH_SIZE)

        for j, emb in enumerate(all_embeddings):
            emb_data["embeddings"].append({
                "id": new_chunks[j]["id"],
                "vector": emb.tolist(),
            })
        print(f"  Done: {len(all_embeddings)} embeddings")

        emb_data["total"] = len(emb_data["embeddings"])

    print("\nSaving...")
    save_state(state)
    save_graph(G)
    save_chunks(chunks_data)
    save_embeddings(emb_data)

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Files processed: {len(new_files)}")
    print(f"  Total chunks: {len(chunks_data['chunks'])}")
    print(f"  Total embeddings: {emb_data['total']}")
    print(f"  Graph nodes: {G.number_of_nodes()}")
    print(f"  Graph edges: {G.number_of_edges()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
