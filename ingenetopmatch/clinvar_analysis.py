"""ClinVar evidence aggregation — port of
``gagi_service/src/data_retrieval/clinvar_analysis.py`` (``analyze_by_clinvar``).

For each selected entity, collect the ClinVar records (``CV(e)`` in the paper) of *other*
variants mapped to the same genomic interval. Production fetches entity coordinates from
Neo4j then bisects a sorted ClinVar table; here the coordinates come from the MiniGraph and
the bisect runs over ``mini_graph/clinvar.csv``.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .graph_client import MiniGraph
from .models import ReportedNeighbor


def analyze_by_clinvar(graph: MiniGraph, neighbors: List[ReportedNeighbor]) -> List[ReportedNeighbor]:
    if not neighbors:
        return neighbors

    entity_ids = [int(n.id) for n in neighbors]
    coords = graph.get_entities_by_ids(entity_ids)

    for neighbor in neighbors:
        entity = coords.get(int(neighbor.id))
        if entity is None:
            raise RuntimeError(f"Coordinates not found for neighbor {neighbor.id}")

        start_interval = int(entity.start_loc)
        end_interval = int(entity.end_loc)
        if start_interval < 0 or end_interval < 0 or start_interval > end_interval:
            raise ValueError(
                f"Invalid coordinate range for neighbor {neighbor.id} "
                f"(start: {start_interval}, end: {end_interval})"
            )

        clinvar_variants = graph.get_clinvar_in_interval(entity.chr, start_interval, end_interval)
        neighbor.clinvar_associations = (
            clinvar_variants if not clinvar_variants.empty else pd.DataFrame()
        )

    return neighbors
