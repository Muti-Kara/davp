"""GWAS evidence aggregation — port of
``gagi_service/src/data_retrieval/gwass_analysis.py`` (``analyze_by_gwass``).

For each selected entity, collect the GWAS evidence (``GWAS(e)`` in the paper) of *other*
variants mapped to the same genomic interval. Production bisects a sorted GWAS table to find
candidate variants, then resolves their relationship ids to
(variant)-[GWAS_association]->(phenotype) records over Neo4j; since the report only uses the
phenotype, the MiniGraph returns those phenotype/disease terms directly. The 1 Mb chunking and
10 Mb interval cap from production are preserved.
"""

from __future__ import annotations

from typing import List

from .graph_client import MiniGraph
from .models import ReportedNeighbor

MAX_INTERVAL_SIZE = 10_000_000  # 10 Mb, matching production
WINDOW_SIZE = 1_000_000  # 1 Mb chunks, matching production


def _iterate_chunks(start: int, end: int, size: int = WINDOW_SIZE):
    cur = start
    while cur <= end:
        yield cur, min(cur + size - 1, end)
        cur += size


def analyze_by_gwas(graph: MiniGraph, neighbors: List[ReportedNeighbor]) -> List[ReportedNeighbor]:
    if not neighbors:
        return neighbors

    entity_ids = [int(n.id) for n in neighbors]
    coords = graph.get_entities_by_ids(entity_ids)

    for neighbor in neighbors:
        entity = coords.get(int(neighbor.id))
        if entity is None:
            raise RuntimeError(f"Entity data missing for neighbor {neighbor.id}")

        start_interval = int(entity.start_loc)
        end_interval = int(entity.end_loc)
        if start_interval < 0 or end_interval < 0 or start_interval > end_interval:
            raise ValueError(
                f"Invalid coordinate range for neighbor {neighbor.id} "
                f"(start: {start_interval}, end: {end_interval})"
            )

        if (end_interval - start_interval) > MAX_INTERVAL_SIZE:
            continue

        for chunk_start, chunk_end in _iterate_chunks(start_interval, end_interval):
            neighbor.gwas_phenotypes.extend(
                graph.get_gwas_phenotypes_in_interval(entity.chr, chunk_start, chunk_end)
            )

    return neighbors
