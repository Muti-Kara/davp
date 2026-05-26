"""MiniGraph — a file-backed stand-in for the GenomicKB Neo4j database.

This class exposes the same query surface that the production ``gagi_service`` reaches for
over Neo4j (``find_chr_chain_by_position``, ``get_chr_chain_with_neighbours``,
``get_entities_by_ids``) plus the sorted-interval overlap lookups production runs alongside the
graph for ClinVar and GWAS. Only the storage backend differs: production answers these with
Cypher against ~3.5e8 nodes; here they are answered from ``mini_graph/graph.json`` +
``clinvar.csv``.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import ChrChain, ChrChainNeighbours, Entity, Phenotype

MINI_GRAPH_DIR = Path(__file__).parent / "mini_graph"


def _normalize_chr(chromosome) -> str:
    """Strip a leading ``chr`` prefix, mirroring production chr normalization."""
    s = str(chromosome)
    if s.lower().startswith("chr"):
        return s[3:]
    return s


def _chromosome_sort_key(chr_val):
    """Sortable chromosome key (1-22, then X, Y, MT, others). Mirrors production."""
    chr_str = _normalize_chr(chr_val).upper()
    if chr_str.isdigit():
        return (0, int(chr_str))
    elif chr_str == "X":
        return (1, 23)
    elif chr_str == "Y":
        return (1, 24)
    elif chr_str == "MT":
        return (1, 25)
    else:
        return (2, hash(chr_str))


class MiniGraph:
    def __init__(
        self,
        chr_chains: List[ChrChain],
        entities: List[Entity],
        phenotypes: Dict[str, Phenotype],
        gwas_variants: List[Dict[str, Any]],
        clinvar_data: pd.DataFrame,
    ):
        self._chr_chains = {c.id: c for c in chr_chains}
        self._entities = {e.id: e for e in entities}
        self._phenotypes = phenotypes
        self.clinvar_data = clinvar_data

        # chr_chain position index: per-chr sorted by start
        self._chains_by_chr: Dict[str, List[ChrChain]] = defaultdict(list)
        for c in chr_chains:
            self._chains_by_chr[_normalize_chr(c.chr)].append(c)
        for chains in self._chains_by_chr.values():
            chains.sort(key=lambda c: c.start_loc)

        # chr_chain -> entities attached to it (an entity attaches to every tile it spans)
        self._entities_by_chain: Dict[int, List[Entity]] = defaultdict(list)
        for e in entities:
            for tile_id in e.props.get("chr_chain_ids", []):
                self._entities_by_chain[tile_id].append(e)

        # ClinVar sorted search keys, mirroring production's combined_keys
        self._clinvar_keys = [
            (_chromosome_sort_key(chrom), int(start))
            for chrom, start in zip(clinvar_data["Chromosome"], clinvar_data["Start"])
        ]

        # GWAS interval-search index: rows of (chr, start_loc, phenotype_id) sorted by (chr, start).
        # Production keeps GWAS variants in a separate sorted table and resolves them to
        # (variant)-[GWAS_association]->(phenotype) over Neo4j; here the phenotype is attached
        # directly because the report only reads the phenotype.
        self._gwas_rows = sorted(
            (
                (_normalize_chr(v["chr"]), int(v["start_loc"]), v["phenotype_id"])
                for v in gwas_variants
            ),
            key=lambda r: (r[0], r[1]),
        )

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, graph_dir: Path | str = MINI_GRAPH_DIR) -> "MiniGraph":
        graph_dir = Path(graph_dir)
        with open(graph_dir / "graph.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        chr_chains = [
            ChrChain(
                id=c["id"],
                chr=str(c["chr"]),
                start_loc=int(c["start_loc"]),
                end_loc=int(c["end_loc"]),
                resolution=int(c.get("resolution", 200)),
                gc_percentage=float(c.get("GC_percentage", 0.0)),
                neighbor_ids=[int(n) for n in c.get("neighbor_ids", [])],
            )
            for c in data["chr_chains"]
        ]
        entities = [
            Entity(
                id=e["id"],
                type=e["type"],
                chr=str(e["chr"]),
                start_loc=int(e["start_loc"]),
                end_loc=int(e["end_loc"]),
                name=e.get("name", ""),
                props=e.get("props", {}),
            )
            for e in data["entities"]
        ]
        phenotypes = {
            p["id"]: Phenotype(id=p["id"], label=p.get("label", ""), definition=p.get("definition", ""))
            for p in data["phenotypes"]
        }
        clinvar_data = cls._load_clinvar(graph_dir / "clinvar.csv")
        return cls(chr_chains, entities, phenotypes, data["gwas_variants"], clinvar_data)

    @staticmethod
    def _load_clinvar(path: Path) -> pd.DataFrame:
        cols = ["Chromosome", "Start", "Stop", "ClinicalSignificance", "PhenotypeList", "PhenotypeIDS"]
        if not path.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(path, dtype={"Chromosome": "str", "PhenotypeList": "str", "PhenotypeIDS": "str"})
        return df.reset_index(drop=True)  # builder already writes it sorted, so bisect works

    # ------------------------------------------------------------------ queries
    # The methods below mirror, one-for-one, the Neo4j helpers in
    # gagi_service/src/neo4j_base/neo4j_utils.py and the interval helpers in
    # data_retrieval/{clinvar,gwass}_analysis.py.

    def find_chr_chain_by_position(self, chromosome, pos: int, resolution: int = 200) -> Optional[ChrChain]:
        """Find the ``chr_chain`` tile spanning a genomic position (FIND_CHR_CHAIN_BY_POSITION)."""
        chrom = _normalize_chr(chromosome)
        for c in self._chains_by_chr.get(chrom, []):
            if c.resolution == resolution and c.start_loc <= pos <= c.end_loc:
                return c
        return None

    def get_chr_chain_neighbours(self, chr_chain_id: int) -> ChrChainNeighbours:
        """Entities + neighbouring tiles attached to a ``chr_chain`` (get_chr_chain_with_neighbours)."""
        entities = list(self._entities_by_chain.get(chr_chain_id, []))
        chain = self._chr_chains.get(chr_chain_id)
        neighbour_chains: List[ChrChain] = []
        if chain is not None:
            for nid in chain.neighbor_ids:
                neighbour = self._chr_chains.get(int(nid))
                if neighbour is not None:
                    neighbour_chains.append(neighbour)
        return ChrChainNeighbours(entities=entities, chr_chains=neighbour_chains)

    def get_entities_by_ids(self, entity_ids) -> Dict[int, Any]:
        """Batch-fetch node coordinates by id (GET_ENTITIES_BY_ID_LIST).

        Production's ``MATCH (n) WHERE id(n) IN $ids`` matches any node type, so this resolves
        both genomic entities and ``chr_chain`` nodes (a selected ``chr_chain`` neighbour carries
        its own interval for ClinVar/GWAS overlap). Both carry ``.chr/.start_loc/.end_loc``.
        """
        out: Dict[int, Any] = {}
        for eid in entity_ids:
            eid = int(eid)
            if eid in self._entities:
                out[eid] = self._entities[eid]
            elif eid in self._chr_chains:
                out[eid] = self._chr_chains[eid]
        return out

    def get_clinvar_in_interval(self, chromosome, start_interval, end_interval) -> pd.DataFrame:
        """ClinVar variants within a genomic interval, via bisect on sorted keys
        (get_clinvar_variants_in_interval)."""
        if self.clinvar_data.empty:
            return pd.DataFrame()
        chr_key = _chromosome_sort_key(chromosome)
        lo = bisect_left(self._clinvar_keys, (chr_key, int(start_interval)))
        hi = bisect_right(self._clinvar_keys, (chr_key, int(end_interval)))
        return self.clinvar_data.iloc[lo:hi].copy()

    def get_gwas_phenotypes_in_interval(self, chromosome, start_interval, end_interval) -> List[Phenotype]:
        """Phenotypes of GWAS variants within a genomic interval, via bisect on the sorted GWAS
        index (production: get_variants_in_interval -> resolve associations). Duplicates are kept,
        one per GWAS hit, so the entity->GWAS mapping repeats a term once per association."""
        chrom = _normalize_chr(chromosome)
        # The 3rd tuple element is the phenotype id; "" and the max code point bound it below
        # and above all real ids, so the bisect ranges purely on (chr, start).
        lo = bisect_left(self._gwas_rows, (chrom, int(start_interval), ""))
        hi = bisect_right(self._gwas_rows, (chrom, int(end_interval), "\U0010ffff"))
        return [self._phenotypes[pid] for (_, _, pid) in self._gwas_rows[lo:hi] if pid in self._phenotypes]
