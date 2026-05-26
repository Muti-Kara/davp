"""inGeneTopMatch — Detailed Variant Report builder for DAVP.

This package is the *real* inGeneTopMatch algorithm described in the DAVP paper
(§ Materials & Methods, "inGeneTopMatch"). In Lidya Genomics' production service it
traverses GenomicKB — a biomedical knowledge graph with ~3.5e8 nodes and ~1.4e9 edges
served from Neo4j — which we cannot redistribute. To make the algorithm runnable outside
production, this package ships:

  * the same traversal / entity-selection / evidence-aggregation / report-assembly logic
    used in production (only the storage backend is swapped from Neo4j to a local file), and
  * a small, synthetic GenomicKB-style graph (``mini_graph/``) that the algorithm runs on.

See ``ingenetopmatch/README.md`` for the limitations and a precise statement of what is
faithful to production and what is a stand-in.
"""

from .models import (
    ChrChain,
    Entity,
    Phenotype,
    GwasAssociation,
    GwasVariant,
    GwasTrio,
    ReportedNeighbor,
    ChrChainNeighbours,
)
from .graph_client import MiniGraph
from .report import build_variant_report, run_ingenetopmatch

__all__ = [
    "ChrChain",
    "Entity",
    "Phenotype",
    "GwasAssociation",
    "GwasVariant",
    "GwasTrio",
    "ReportedNeighbor",
    "ChrChainNeighbours",
    "MiniGraph",
    "build_variant_report",
    "run_ingenetopmatch",
]
