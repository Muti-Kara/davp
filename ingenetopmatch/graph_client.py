"""MiniGraph — a file-backed stand-in for the GenomicKB Neo4j database.

This class exposes the same query surface that the production ``gagi_service`` reaches for
over Neo4j (``find_chr_chain_by_position``, ``get_chr_chain_with_neighbours``,
``get_entities_by_ids``, ``get_gwas_associations_by_rel_ids``) plus the two sorted-interval
tables production keeps alongside the graph for fast overlap search (ClinVar and GWAS).
Only the storage backend differs: production answers these with Cypher against ~3.5e8 nodes;
here they are answered from ``mini_graph/graph.json`` + ``clinvar.csv`` + ``gwas.csv``.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import (
    ChrChain,
    ChrChainNeighbours,
    Entity,
    GwasAssociation,
    GwasTrio,
    GwasVariant,
    Phenotype,
)

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
        gwas_variants: List[GwasVariant],
        gwas_associations: Dict[int, GwasAssociation],
        clinvar_data: pd.DataFrame,
    ):
        self._chr_chains = {c.id: c for c in chr_chains}
        self._entities = {e.id: e for e in entities}
        self._phenotypes = phenotypes
        self._gwas_variants = {v.id: v for v in gwas_variants}
        self._gwas_associations = gwas_associations
        self.clinvar_data = clinvar_data

        # chr_chain position index: per-chr sorted (start_loc, chr_chain)
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

        # GWAS interval-search table: rows of (chr, start_loc, rel_id) sorted by (chr, start)
        self._gwas_rows = sorted(
            ((_normalize_chr(v.chr), v.start_loc, v.rel_id) for v in gwas_variants),
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
                sequence=c.get("sequence", ""),
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
            p["id"]: Phenotype(
                id=p["id"],
                label=p.get("label", ""),
                definition=p.get("definition", ""),
                type=p.get("type", ""),
            )
            for p in data["phenotypes"]
        }
        gwas_variants = [
            GwasVariant(
                id=v["id"],
                chr=str(v["chr"]),
                start_loc=int(v["start_loc"]),
                rel_id=int(v["rel_id"]),
                phenotype_id=v["phenotype_id"],
            )
            for v in data["gwas_variants"]
        ]
        gwas_associations = {
            int(v["rel_id"]): GwasAssociation(
                rel_id=int(v["rel_id"]),
                risk_allele=v.get("risk_allele", ""),
                mlog_pvalue=float(v.get("mlog_pvalue", 0.0)),
                pubmed_id=v.get("pubmed_id"),
                accession=v.get("accession", ""),
            )
            for v in data["gwas_variants"]
        }

        clinvar_data = cls._load_clinvar(graph_dir / "clinvar.csv")
        return cls(chr_chains, entities, phenotypes, gwas_variants, gwas_associations, clinvar_data)

    @staticmethod
    def _load_clinvar(path: Path) -> pd.DataFrame:
        cols = ["Chromosome", "Start", "Stop", "ClinicalSignificance", "PhenotypeList", "PhenotypeIDS"]
        if not path.exists():
            return pd.DataFrame(columns=cols)
        df = pd.read_csv(path, dtype={"Chromosome": "str", "PhenotypeList": "str", "PhenotypeIDS": "str"})
        # Keep sorted by (chromosome key, Start) so bisect works (builder writes it sorted).
        return df.reset_index(drop=True)

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
            # Explicit chr_chain edges (e.g. links to coarser region tiles), as in GenomicKB.
            for nid in chain.neighbor_ids:
                neighbour = self._chr_chains.get(int(nid))
                if neighbour is not None:
                    neighbour_chains.append(neighbour)
            # Fall back to positional adjacency if no explicit edges are recorded.
            if not chain.neighbor_ids:
                for c in self._chains_by_chr.get(_normalize_chr(chain.chr), []):
                    if c.id != chain.id and (
                        c.end_loc + 1 == chain.start_loc or chain.end_loc + 1 == c.start_loc
                    ):
                        neighbour_chains.append(c)
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

    def get_gwas_rel_ids_in_interval(self, chromosome, start_interval, end_interval) -> List[int]:
        """Relationship ids of GWAS variants within an interval, via bisect (get_variants_in_interval)."""
        chrom = _normalize_chr(chromosome)
        lo = bisect_left(self._gwas_rows, (chrom, int(start_interval), -1))
        hi = bisect_right(self._gwas_rows, (chrom, int(end_interval), float("inf")))
        return [rel_id for (_, _, rel_id) in self._gwas_rows[lo:hi]]

    def get_gwas_associations_by_rel_ids(self, rel_ids) -> List[GwasTrio]:
        """Resolve (variant)-[GWAS_association]->(phenotype) trios by relationship id
        (GET_GWAS_ASSOCIATIONS_BY_REL_IDS)."""
        trios: List[GwasTrio] = []
        # rel_id == GwasVariant.id in this mini-graph (one association per variant node)
        rel_to_variant = {v.rel_id: v for v in self._gwas_variants.values()}
        for rel_id in rel_ids:
            rel_id = int(rel_id)
            variant = rel_to_variant.get(rel_id)
            if variant is None:
                continue
            phenotype = self._phenotypes.get(variant.phenotype_id)
            if phenotype is None:
                continue
            trios.append(
                GwasTrio(
                    variant=variant,
                    association=self._gwas_associations[rel_id],
                    phenotype=phenotype,
                )
            )
        return trios
