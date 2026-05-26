"""Data models for the mini GenomicKB graph and inGeneTopMatch evidence.

These mirror the production ``gagi_service`` models (genes, transcripts, regulatory
elements, ``chr_chain`` tiles, phenotype/disease ontology nodes) but are flattened for a
JSON-backed graph. Production wraps every GenomicKB node type in its own typed dataclass
(``Gene``, ``Transcript``, ``TFBindingSite``, ...); here a single ``Entity`` carries the node
``type`` label so the selection logic stays identical while the model layer stays small.

The model is trimmed to exactly what the Detailed Variant Report uses. In particular, GWAS
evidence is represented by the ``phenotype_or_disease`` nodes reached from a variant — the
production ``GwassTrio`` additionally materialises the GWAS variant node and the relationship
properties, but the report only ever reads the phenotype, so this port collects phenotypes
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ChrChain:
    """A 200 bp ``chr_chain`` segment tiling the genome (GenomicKB ``chr_chain`` node).

    ``neighbor_ids`` are the ids of ``chr_chain`` nodes linked to this one (GenomicKB stores
    explicit chr_chain--chr_chain edges, including links to coarser-resolution region tiles).
    """

    id: int
    chr: str
    start_loc: int
    end_loc: int
    resolution: int = 200
    gc_percentage: float = 0.0
    neighbor_ids: List[int] = field(default_factory=list)


@dataclass
class Entity:
    """A genomic entity overlapping the genome (gene, transcript, regulatory element, ...).

    ``type`` is the human-readable label used in the Detailed Variant Report
    (e.g. ``"Gene"``, ``"Transcript"``, ``"TF Binding Site"``, ``"ChromHMM State"``),
    matching the labels produced by the production entity selector.
    """

    id: int
    type: str
    chr: str
    start_loc: int
    end_loc: int
    name: str = ""
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phenotype:
    """A ``phenotype_or_disease`` ontology node (EFO / HP / GO / MONDO ... term)."""

    id: str
    label: str
    definition: str = ""


@dataclass
class ReportedNeighbor:
    """A genomic entity selected as overlapping the query variant, enriched with the GWAS and
    ClinVar evidence collected from other variants mapped to it.

    ``gwas_phenotypes`` is the list of phenotype/disease terms reached via GWAS associations of
    other variants in this entity's interval (duplicates kept — the entity->GWAS mapping repeats
    a term once per association). ``clinvar_associations`` is a pandas DataFrame so the
    report-assembly code matches production.
    """

    type: str
    id: int
    gwas_phenotypes: List[Phenotype] = field(default_factory=list)
    clinvar_associations: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ChrChainNeighbours:
    """Entities and neighbouring tiles attached to a ``chr_chain`` node.

    Production splits these into ~16 typed lists; the selection criterion is identical
    across types, so here they are a single ``entities`` list plus the neighbouring
    ``chr_chains`` (which production also tests for overlap separately)."""

    entities: List[Entity] = field(default_factory=list)
    chr_chains: List[ChrChain] = field(default_factory=list)
