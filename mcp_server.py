#!/usr/bin/env python3
"""
South Indian Inscriptions Graph RAG — MCP Server

Exposes the inscriptions knowledge graph as MCP tools for Claude Code.
Covers 4,500+ pages of South Indian epigraphic records — dynasties, kings,
temples, deities, locations, dates, and their relationships.
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

# --- Config ---
DATA_DIR = Path(__file__).parent / "data"
GRAPH_FILE = DATA_DIR / "graph.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"

TOP_K = 10

# --- Global state (loaded once at startup) ---
G = None
CHUNKS = None
EMB_IDS = None
EMB_MATRIX = None
EMBED_MODEL = None


def ensure_loaded():
    """Lazy-load data and model on first use."""
    global G, CHUNKS, EMB_IDS, EMB_MATRIX, EMBED_MODEL

    if G is not None:
        return

    sys.stderr.write("Loading inscriptions graph data...\n")

    with open(GRAPH_FILE) as f:
        G = nx.node_link_graph(json.load(f))

    with open(CHUNKS_FILE) as f:
        chunks_data = json.load(f)
    CHUNKS = {c["id"]: c for c in chunks_data["chunks"]}

    with open(EMBEDDINGS_FILE) as f:
        emb_data = json.load(f)
    EMB_IDS = [e["id"] for e in emb_data["embeddings"]]
    EMB_MATRIX = np.array([e["vector"] for e in emb_data["embeddings"]])

    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    sys.stderr.write(f"Loaded: {G.number_of_nodes()} nodes, {len(CHUNKS)} chunks, {len(EMB_IDS)} embeddings\n")


def vector_search(query_text, top_k=TOP_K):
    """Search chunks by semantic similarity."""
    ensure_loaded()

    query_vec = EMBED_MODEL.encode([query_text])[0]
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = EMB_MATRIX / np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True)
    similarities = matrix_norm @ query_norm
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        chunk_id = EMB_IDS[idx]
        chunk = CHUNKS.get(chunk_id, {})
        results.append({
            "score": round(float(similarities[idx]), 4),
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", "")[:1500],
        })

    return results


def graph_expand(sources):
    """Get entities connected to source documents."""
    ensure_loaded()
    entities = {}

    for source in sources:
        if not G.has_node(source):
            continue
        for neighbor in G.predecessors(source):
            data = dict(G.nodes[neighbor])
            if data.get("type") == "document":
                continue
            name = data.get("name", neighbor)
            if name not in entities:
                entities[name] = {
                    "type": data.get("type", "unknown"),
                    "description": data.get("description", ""),
                    "mentions": data.get("mentions", 0),
                    "connections": [],
                }
            for nn in G.neighbors(neighbor):
                if G.nodes[nn].get("type") != "document":
                    nn_data = dict(G.nodes[nn])
                    edge_data = G.edges[neighbor, nn]
                    entities[name]["connections"].append({
                        "name": nn_data.get("name", nn),
                        "relationship": edge_data.get("type", "related_to"),
                    })

    return entities


def search_inscriptions(query, top_k=TOP_K):
    """Full Graph RAG search: vector search + graph expansion."""
    chunks = vector_search(query, top_k)
    sources = list({c["source"] for c in chunks if c["source"]})
    entities = graph_expand(sources)

    parts = ["## Retrieved Inscriptions Context\n"]
    for c in chunks:
        parts.append(f"**{c['title']}** ({c['source']})\nRelevance: {c['score']}\n{c['text']}\n\n---\n")

    if entities:
        parts.append("\n## Related Historical Entities\n")
        for name, data in sorted(entities.items(), key=lambda x: -x[1]["mentions"]):
            conns = ", ".join([f"{c['name']} ({c['relationship']})" for c in data["connections"][:5]])
            parts.append(f"- **{name}** ({data['type']}, {data['mentions']}x): {data['description']}. Connections: {conns}")

    return "\n".join(parts)


def get_entity(name):
    """Get full info about a historical entity."""
    ensure_loaded()

    node_id = name.lower().replace(" ", "_")
    if not G.has_node(node_id):
        matches = [n for n in G.nodes if name.lower() in n.lower()]
        if not matches:
            return f"Entity '{name}' not found. Try a different name or spelling."
        node_id = matches[0]

    data = dict(G.nodes[node_id])
    info_parts = [
        f"# {data.get('name', node_id)}",
        f"Type: {data.get('type', 'unknown')}",
        f"Description: {data.get('description', 'N/A')}",
        f"Mentioned {data.get('mentions', 0)} times",
        f"Found in: {', '.join(data.get('sources', [])[:10])}",
        "\n## Connections:",
    ]

    for neighbor in G.neighbors(node_id):
        nn_data = dict(G.nodes[neighbor])
        edge_data = G.edges[node_id, neighbor]
        if nn_data.get("type") != "document":
            info_parts.append(
                f"- → {nn_data.get('name', neighbor)} ({edge_data.get('type', 'related_to')}): {edge_data.get('context', '')}"
            )

    for pred in G.predecessors(node_id):
        nn_data = dict(G.nodes[pred])
        edge_data = G.edges[pred, node_id]
        if nn_data.get("type") != "document" and pred != node_id:
            info_parts.append(
                f"- ← {nn_data.get('name', pred)} ({edge_data.get('type', 'related_to')}): {edge_data.get('context', '')}"
            )

    return "\n".join(info_parts)


def list_entities(entity_type=None):
    """List entities, optionally filtered by type."""
    ensure_loaded()

    entities = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "document":
            continue
        if entity_type and entity_type != "all" and data.get("type") != entity_type:
            continue
        entities.append({
            "name": data.get("name", node_id),
            "type": data.get("type", "unknown"),
            "mentions": data.get("mentions", 0),
        })

    entities.sort(key=lambda x: -x["mentions"])
    lines = [f"## Inscription Entities ({entity_type or 'all'})\n"]
    for e in entities[:100]:
        lines.append(f"- **{e['name']}** ({e['type']}, {e['mentions']}x)")
    return "\n".join(lines)


# --- MCP Protocol Implementation (stdio JSON-RPC) ---

def handle_request(request):
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "inscriptions-graph-rag", "version": "1.0.0"},
            }
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "search_inscriptions",
                        "description": "Search the South Indian Inscriptions archive (4,500+ pages covering Pallava, Chola, Chalukya, Pandya, Rashtrakuta, Hoysala, Vijayanagara dynasties and more) using Graph RAG. Returns relevant inscription text chunks plus related historical entities from the knowledge graph.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Natural language search query about South Indian history, inscriptions, dynasties, temples, rulers, etc."},
                                "top_k": {"type": "integer", "description": "Number of results (default 10)", "default": 10},
                            },
                            "required": ["query"],
                        }
                    },
                    {
                        "name": "get_inscription_entity",
                        "description": "Get detailed info about a historical entity (dynasty, ruler, temple, deity, location, inscription) from the South Indian Inscriptions knowledge graph. Shows connections, mentions, and source documents.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Entity name (e.g., 'Rajasimha', 'Pallava', 'Kanchipuram', 'Siva', 'Mamallapuram')"},
                            },
                            "required": ["name"],
                        }
                    },
                    {
                        "name": "list_inscription_entities",
                        "description": "List all historical entities in the South Indian Inscriptions knowledge graph, optionally filtered by type.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Entity type filter: person, dynasty, temple, deity, location, admin_division, inscription, date, land_grant, tax_revenue, currency_measure, title, religious_order, guild_assembly, ritual_festival, script_language, or 'all'", "default": "all"},
                            },
                        }
                    },
                ]
            }
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            if tool_name == "search_inscriptions":
                result = search_inscriptions(tool_args["query"], tool_args.get("top_k", TOP_K))
            elif tool_name == "get_inscription_entity":
                result = get_entity(tool_args["name"])
            elif tool_name == "list_inscription_entities":
                result = list_entities(tool_args.get("type", "all"))
            else:
                result = f"Unknown tool: {tool_name}"

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result}]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True,
                }
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }


def main():
    """Run MCP server over stdio."""
    sys.stderr.write("South Indian Inscriptions Graph RAG MCP server starting...\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
