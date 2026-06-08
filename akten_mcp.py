#!/usr/bin/env python3
"""
MCP server exposing the donkredito Ermittlungsakten retriever.

Transport: stdio. Tool: `search_akten`.

Run directly:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python akten_mcp.py

Wire into another repo (Claude Code / Desktop) by adding to its MCP config:

    {
      "mcpServers": {
        "donkredito-akten": {
          "command": "/abs/path/to/RAG_LW/.venv/bin/python",
          "args": ["/abs/path/to/RAG_LW/akten_mcp.py"],
          "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"
          }
        }
      }
    }

Requires .env.local in the working directory with QDRANT_ENDPOINT + QDRANT_API_KEY.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load env from the repo root (where this file lives), regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env.local", override=True)

# Default to offline mode — the mxbai-de model is gated on HF but cached locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# All logging must go to stderr — stdout is the MCP protocol channel.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("akten-mcp")

# Imports must come AFTER env setup (they trigger model load on first call).
sys.path.insert(0, str(REPO_ROOT))
from mcp.server.fastmcp import FastMCP  # noqa: E402
from ours_mxbai_client import OursMxbaiRetriever  # noqa: E402

COLLECTION = "ermittlungsakten_mxbai"

mcp = FastMCP("donkredito-akten")

_RETRIEVER: Optional[OursMxbaiRetriever] = None


def _retriever() -> OursMxbaiRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        log.info("Lazy-loading OursMxbaiRetriever (collection=%s)...", COLLECTION)
        _RETRIEVER = OursMxbaiRetriever(collection=COLLECTION)
    return _RETRIEVER


@mcp.tool()
def search_akten(
    query: str,
    top_k: int = 5,
    band: Optional[str] = None,
    dokument_typ: Optional[str] = None,
    page_min: Optional[int] = None,
    page_max: Optional[int] = None,
    expand: bool = True,
    rerank: bool = True,
) -> list[dict]:
    """
    Durchsucht die Ermittlungsakten zum Fall "donkredito" (Az. 107 Js 1286/20,
    StA Aachen — gewerbsmäßiger Bandenbetrug via donkredito.com).

    Liefert relevante Textpassagen aus drei Aktenbänden (HA Band I–III) mit
    Seitenreferenz für direkte Zitierung. Die Pipeline nutzt Query Expansion,
    Hybrid-Suche (Dense mxbai-de + BM25) und Cohere-Reranking.

    Args:
        query: Suchanfrage in natürlicher Sprache (Deutsch).
        top_k: Anzahl Treffer (1–20, default 5).
        band: Optional auf einen Aktenband eingrenzen ("I", "II" oder "III").
        dokument_typ: Optional auf einen Dokumenttyp filtern. Mögliche Werte:
            "Anklageschrift", "Zeugenaussage", "Vermerk", "Beschluss",
            "Schreiben", "Abrechnung", "Durchsuchung", "Haftbefehl",
            "Protokoll", "Gutachten", "Urteil", "Bewertung", "Sonstiges".
        page_min: Untere Seitenzahl-Grenze (inklusive).
        page_max: Obere Seitenzahl-Grenze (inklusive).
        expand: Query-Expansion vor der Suche (default True).
        rerank: Cohere-Reranking der Kandidaten (default True).

    Returns:
        Liste von Treffern. Jeder Treffer ist ein Dictionary mit:
          - text          Chunk-Volltext (Markdown, "## Seite N"-Marker
                          inline für mehrseitige Chunks)
          - score         Relevanz-Score (0–1, höher = besser)
          - fall          "donkredito"
          - band          "I", "II" oder "III"
          - page_start    Erste Seitennummer dieses Chunks
          - page_end      Letzte Seitennummer dieses Chunks
          - page_label    Formatierter Seitenbereich ("S. 18" / "S. 18–20")
          - aktenzeichen  "107 Js 1286/20"
          - dokument_typ  Klassifikation
          - source_file   Pfad der Quell-Datei im Repo
          - source_id     Datei-Stem ("echte_akte1", "echte_akte2", "echte_akte3")
    """
    top_k = max(1, min(int(top_k), 20))

    band_norm = band.strip().upper() if band else None
    valid_bands = {"I", "II", "III"}
    if band_norm and band_norm not in valid_bands:
        raise ValueError(f"band muss eines von {sorted(valid_bands)} sein, war: {band!r}")

    has_post_filter = band_norm is not None or page_min is not None or page_max is not None
    fetch_k = top_k * 5 if has_post_filter else top_k

    retr = _retriever()._retriever  # underlying JuristischerRetriever
    results, _ = retr.search(
        query,
        collections=[COLLECTION],
        top_k=fetch_k,
        dokument_typ=dokument_typ,
        expand=expand,
        rerank=rerank,
    )

    out: list[dict] = []
    for r in results:
        m = r.metadata or {}
        if band_norm and m.get("band") != band_norm:
            continue
        ps = m.get("page_start")
        if page_min is not None and (ps is None or ps < page_min):
            continue
        if page_max is not None and (ps is None or ps > page_max):
            continue

        pe = m.get("page_end")
        if ps is not None and pe is not None and pe != ps:
            page_label = f"S. {ps}–{pe}"
        elif ps is not None:
            page_label = f"S. {ps}"
        else:
            page_label = ""

        out.append({
            "text": r.text,
            "score": float(r.score),
            "fall": m.get("fall", ""),
            "band": m.get("band", ""),
            "page_start": ps,
            "page_end": pe,
            "page_label": page_label,
            "aktenzeichen": m.get("aktenzeichen", ""),
            "dokument_typ": m.get("dokument_typ", ""),
            "source_file": m.get("source_file", ""),
            "source_id": m.get("source_id", ""),
        })
        if len(out) >= top_k:
            break

    return out


if __name__ == "__main__":
    log.info("Starte donkredito-akten MCP-Server (stdio)...")
    mcp.run()
