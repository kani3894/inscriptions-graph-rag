#!/usr/bin/env python3
"""
South Indian Inscriptions Graph RAG — Query CLI

Usage:
    python3 query.py "Pallava temples at Mamallapuram"
    python3 query.py --entity "Rajasimha"
    python3 query.py --list-entities dynasty
    python3 query.py --stats
"""

import json
import sys
import argparse
from pathlib import Path

import networkx as nx
import numpy as np

# --- Config ---
DATA_DIR = Path(__file__).parent / "data"
GRAPH_FILE = DATA_DIR / "graph.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"

TOP_K = 10
GRAPH_HOPS = 1


def load_data():
    """Load graph, chunks, and embeddings."""
    with open(GRAPH_FILE) as f:
        G = nx.node_link_graph(json.load(f))

    with open(CHUNKS_FILE) as f:
        chunks_data = json.load(f)
    chunks = {c["id"]: c for c in chunks_data["chunks"]}

    with open(EMBEDDINGS_FILE) as f:
        emb_data = json.load(f)
    emb_ids = [e["id"] for e in emb_data["embeddings"]]
    emb_matrix = np.array([e["vector"] for e in emb_data["embeddings"]])

    return G, chunks, emb_ids, emb_matrix


def embed_query(text):
    """Embed query text locally."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text])[0]


def search_chunks(query_vec, emb_ids, emb_matrix, top_k=TOP_K):
    """Cosine similarity search."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    similarities = matrix_norm @ query_norm
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "id": emb_ids[idx],
            "score": float(similarities[idx]),
        })
    return results


def expand_graph(G, chunks, chunk_results, hops=GRAPH_HOPS):
    """For retrieved chunks, find related entities via graph traversal."""
    sources = set()
    for r in chunk_results:
        chunk = chunks.get(r["id"], {})
        if chunk.get("source"):
            sources.add(chunk["source"])

    related_entities = {}
    for source in sources:
        if not G.has_node(source):
            continue
        for neighbor in G.predecessors(source):
            if G.nodes[neighbor].get("type") != "document":
                node_data = dict(G.nodes[neighbor])
                name = node_data.get("name", neighbor)
                if name not in related_entities:
                    related_entities[name] = node_data
                    related_entities[name]["connections"] = []

                for nn in G.neighbors(neighbor):
                    if nn != source and G.nodes[nn].get("type") != "document":
                        nn_data = dict(G.nodes[nn])
                        edge_data = G.edges[neighbor, nn]
                        related_entities[name]["connections"].append({
                            "name": nn_data.get("name", nn),
                            "type": nn_data.get("type", "unknown"),
                            "relationship": edge_data.get("type", "related_to"),
                        })

    return related_entities


def get_entity_info(G, entity_name):
    """Get everything about a specific entity from the graph."""
    node_id = entity_name.lower().replace(" ", "_")

    if not G.has_node(node_id):
        matches = [n for n in G.nodes if entity_name.lower() in n.lower()]
        if not matches:
            return None
        node_id = matches[0]

    node_data = dict(G.nodes[node_id])
    info = {
        "name": node_data.get("name", node_id),
        "type": node_data.get("type", "unknown"),
        "description": node_data.get("description", ""),
        "mentions": node_data.get("mentions", 0),
        "sources": node_data.get("sources", []),
        "connections": [],
    }

    for neighbor in G.neighbors(node_id):
        nn_data = dict(G.nodes[neighbor])
        edge_data = G.edges[node_id, neighbor]
        info["connections"].append({
            "name": nn_data.get("name", neighbor),
            "type": nn_data.get("type", "unknown"),
            "relationship": edge_data.get("type", "related_to"),
            "context": edge_data.get("context", ""),
        })

    for predecessor in G.predecessors(node_id):
        if predecessor == node_id:
            continue
        nn_data = dict(G.nodes[predecessor])
        edge_data = G.edges[predecessor, node_id]
        info["connections"].append({
            "name": nn_data.get("name", predecessor),
            "type": nn_data.get("type", "unknown"),
            "relationship": edge_data.get("type", "related_to") + " (incoming)",
            "context": edge_data.get("context", ""),
        })

    return info


def list_entities(G, entity_type=None):
    """List all entities, optionally filtered by type."""
    entities = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == "document":
            continue
        if entity_type and data.get("type") != entity_type:
            continue
        entities.append({
            "name": data.get("name", node_id),
            "type": data.get("type", "unknown"),
            "mentions": data.get("mentions", 0),
            "description": data.get("description", ""),
        })

    return sorted(entities, key=lambda x: x["mentions"], reverse=True)


def format_results(query, chunks, chunk_results, graph_context):
    """Format results for display."""
    parts = [f"## Results for: {query}\n"]

    for r in chunk_results:
        chunk = chunks.get(r["id"], {})
        source = chunk.get("source", "unknown")
        title = chunk.get("title", "")
        header = f"**{title or source}**"
        header += f" — score: {r['score']:.3f}"
        parts.append(f"{header}\nSource: {source}\n{chunk.get('text', '')}\n\n---\n")

    if graph_context:
        parts.append("\n## Related Entities\n")
        for name, data in graph_context.items():
            desc = data.get("description", "")
            etype = data.get("type", "")
            conns = data.get("connections", [])
            conn_str = ", ".join([f"{c['name']} ({c['relationship']})" for c in conns[:5]])
            parts.append(f"- **{name}** ({etype}): {desc}. Connected to: {conn_str}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="South Indian Inscriptions Graph RAG Query")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--entity", help="Get info about a specific entity")
    parser.add_argument("--list-entities", metavar="TYPE", nargs="?", const="all",
                        help="List entities (filter by type: person, dynasty, temple, location, deity, inscription, date, concept, title)")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    G, chunks, emb_ids, emb_matrix = load_data()

    if args.stats:
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        print(f"Chunks: {len(chunks)}")
        print(f"Embeddings: {len(emb_ids)}")
        type_counts = {}
        for _, data in G.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")
        return

    if args.list_entities:
        etype = None if args.list_entities == "all" else args.list_entities
        entities = list_entities(G, etype)
        if args.json:
            print(json.dumps(entities, indent=2))
        else:
            for e in entities[:50]:
                print(f"  [{e['type']}] {e['name']} (mentioned {e['mentions']}x) — {e['description']}")
        return

    if args.entity:
        info = get_entity_info(G, args.entity)
        if not info:
            print(f"Entity '{args.entity}' not found in graph.")
            return
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"\n{info['name']} ({info['type']})")
            print(f"  Description: {info['description']}")
            print(f"  Mentioned {info['mentions']} times across {len(info['sources'])} documents")
            print(f"\n  Connections:")
            for c in info["connections"][:20]:
                print(f"    → {c['name']} ({c['relationship']}) {c.get('context', '')}")
        return

    if not args.query:
        parser.print_help()
        return

    print(f"Searching: {args.query}\n")
    query_vec = embed_query(args.query)
    chunk_results = search_chunks(query_vec, emb_ids, emb_matrix, args.top_k)
    graph_context = expand_graph(G, chunks, chunk_results)

    if args.json:
        print(json.dumps({
            "chunks": [{"id": r["id"], "score": r["score"], "text": chunks.get(r["id"], {}).get("text", "")[:200]}
                        for r in chunk_results],
            "entities": {k: {"type": v.get("type"), "connections": len(v.get("connections", []))}
                         for k, v in graph_context.items()},
        }, indent=2))
    else:
        output = format_results(args.query, chunks, chunk_results, graph_context)
        print(output)


if __name__ == "__main__":
    main()
