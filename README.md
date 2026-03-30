# South Indian Inscriptions — Graph RAG

A knowledge graph and retrieval-augmented generation (RAG) system built over 4,567 digitized South Indian epigraphic records. Extracts historical entities (dynasties, kings, temples, deities, locations, dates) and their relationships, then exposes them as searchable tools via an MCP server for Claude Code.

## What's in here

- **`ingest.py`** — Ingestion pipeline. Scans markdown inscription files, extracts entities using LLM (OpenRouter / ollama qwen3), builds a NetworkX graph, and embeds text chunks with `all-MiniLM-L6-v2`.
- **`mcp_server.py`** — MCP server exposing `search_knowledge`, `get_entity`, and `list_entities` tools for Claude Code.
- **`query.py`** — CLI for searching the graph and exploring entities.
- **`data/`** — Graph (`10,638 nodes / 15,092 edges`), chunks (`7,437`), and embeddings.

## Current state

- 4,567 inscription files ingested and chunked
- ~1,016 files have full entity extraction; ~2,800 remaining (ran out of OpenRouter credits mid-extraction)
- ~650 entities need type reclassification (see `TODO-entity-cleanup.md`)

## TODO — Entity cleanup & extraction completion

See `TODO-entity-cleanup.md` for details. Summary:
1. Reclassify ~650 mistyped entities (merge `king`→`person`, `village`→`location`, use LLM for `unknown`/`concept` types)
2. Complete entity extraction on remaining ~2,800 files using Gemini free API (1,500 calls/day)

## Stack

Python, NetworkX, sentence-transformers (all-MiniLM-L6-v2), OpenRouter/ollama for entity extraction, MCP protocol
