"""Algorithmic entity selection — port of
``gagi_service/src/services/algorithmic_entity_selector.py``.

Selects the genomic entities overlapping the variant position. This is the
``E_i = {e in V : pos(v_i) in span(e)}`` step from the paper. The selection criterion is
identical to production: direct overlap (``start <= pos <= end``) or proximity within a
+/-10 bp window; neighbouring ``chr_chain`` tiles are tested for direct overlap only.
"""

from __future__ import annotations

from typing import List

from .models import ChrChainNeighbours, ReportedNeighbor

PROXIMITY_BP = 10  # +/- window, matching production


def select_entities_algorithmically(
    variant_chr: str, variant_pos: int, neighbours: ChrChainNeighbours
) -> List[ReportedNeighbor]:
    # Normalize chromosome (drop 'chr' prefix) for comparison, as in production.
    vchr = str(variant_chr)
    if vchr.lower().startswith("chr"):
        vchr = vchr[3:]

    selected: List[ReportedNeighbor] = []

    # Production iterates ~14 typed neighbour lists and applies this same test to each;
    # here every candidate carries its own ``type`` label, so one loop suffices.
    for entity in neighbours.entities:
        if str(entity.chr) != vchr:
            continue
        overlap = entity.start_loc <= variant_pos <= entity.end_loc
        proximal = (
            abs(entity.start_loc - variant_pos) <= PROXIMITY_BP
            or abs(entity.end_loc - variant_pos) <= PROXIMITY_BP
        )
        if overlap or proximal:
            selected.append(ReportedNeighbor(type=entity.type, id=entity.id))

    # chr_chain tiles: direct overlap only (no proximity window), matching production.
    for chain in neighbours.chr_chains:
        if str(chain.chr) != vchr:
            continue
        if chain.start_loc <= variant_pos <= chain.end_loc:
            selected.append(ReportedNeighbor(type="chr_chain", id=chain.id))

    if not selected:
        raise RuntimeError(
            f"No genomic entities found overlapping variant at {vchr}:{variant_pos}. "
            f"This may indicate insufficient graph coverage for this region."
        )

    return selected
