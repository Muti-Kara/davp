from pydantic import BaseModel, Field, create_model
from typing import Literal, Union
from abc import ABC, abstractmethod
import random
import copy
import math

from llm.session import Session


# Models
class CmpItem:
    def __init__(self, obj_id: str, obj_info: str, rank: int = 0):
        self.obj_id = obj_id
        self.obj_info = obj_info
        self.rank = rank
        
        self.lost_to = []
        self.won_against = []


class Elimin8Item(CmpItem):
    def __init__(self, obj_id: str, obj_info: str, rank: int = 0, score: int = 0):
        super().__init__(obj_id, obj_info, rank)
        self.score = score


class RoundRobinItem(CmpItem):
    def __init__(self, obj_id: str, obj_info: str, rank: int = 0, wins: int = 0, score_diff: int = 0):
        super().__init__(obj_id, obj_info, rank)
        self.wins = wins
        self.score_diff = score_diff


class RankingItem(BaseModel):
    id: str
    description: str
    
    def to_tuple(self) -> tuple[str, str]:
        return (self.id, self.description)
    
    def to_cmp_item(self, rank: int = 0) -> CmpItem:
        return CmpItem(obj_id=self.id, obj_info=self.description, rank=rank)


class RankingInput(BaseModel):
    items: list[RankingItem]
    
    def to_cmp_items(self) -> list[CmpItem]:
        return [item.to_cmp_item(rank=i) for i, item in enumerate(self.items)]


class RankingResult(BaseModel):
    id: str
    description: str
    rank: int
    
    @classmethod
    def from_tuple(cls, item: tuple[str, str], rank: int) -> "RankingResult":
        return cls(id=item[0], description=item[1], rank=rank)


class Elimin8Comparison(BaseModel):
    group_items: list[str] = Field(description="IDs of all 8 items in the group")
    rankings: list[str] = Field(description="List of item_ids in the order of ranking (1-8)")


class RoundRobinComparison(BaseModel):
    item_1_id: str
    item_2_id: str
    winner_id: str
    score_difference: int


Comparison = Union[Elimin8Comparison, RoundRobinComparison]


class RankingSession(BaseModel):
    results: list[RankingResult]
    comparison_history: list[Comparison] = Field(default_factory=list)
    round_history: list[list[str]] = Field(default_factory=list)

    @classmethod
    def from_tuples(cls, ranked_tuples: list[tuple[str, str]], history: list[Comparison], round_history: list[list[str]] = None) -> "RankingSession":
        results = [RankingResult.from_tuple(item, rank=i+1) for i, item in enumerate(ranked_tuples)]
        return cls(results=results, comparison_history=history, round_history=round_history or [])


class Elimin8Response(BaseModel):
    rank_1: str = Field(description="Item identifier")
    rank_2: str = Field(description="Item identifier")
    rank_3: str = Field(description="Item identifier")
    rank_4: str = Field(description="Item identifier")
    rank_5: str = Field(description="Item identifier")
    rank_6: str = Field(description="Item identifier")
    rank_7: str = Field(description="Item identifier")
    rank_8: str = Field(description="Item identifier")

    def to_list(self) -> list[str]:
        return [self.rank_1, self.rank_2, self.rank_3, self.rank_4, self.rank_5, self.rank_6, self.rank_7, self.rank_8]


class RoundRobinMatchResult(BaseModel):
    winner_id: str = Field(description="Winner identifier")
    score_difference: int = Field(description="Score difference between the two items")


# Comparators
class BaseComparator(ABC):
    def __init__(self, session: Session, template: str, required_fields: list[str]):
        self.session = session
        self.template = template
        self.comparison_history: list = []
        
        self._validate_template(required_fields)
    
    def _validate_template(self, required_fields: list[str]):
        for field in required_fields:
            if f"{{{field}}}" not in self.template:
                raise ValueError(f"Template missing required field: {{{field}}}")

        try:
            test_values = {field: f"{field}_test" for field in required_fields}
            formatted = self.template.format(**test_values)
            for field in required_fields:
                if f"{field}_test" not in formatted:
                    raise ValueError(f"Field {{{field}}} not properly formatted")
        except KeyError as e:
            raise ValueError(f"Template validation failed: {e}")
    
    @abstractmethod
    def batch_compare(self, *args, **kwargs):
        pass


class Elimin8Comparator(BaseComparator):
    def __init__(self, session: Session, template: str):
        super().__init__(session, template, ["target"] + [f"item_{i+1}_id" for i in range(8)] + [f"item_{i+1}_info" for i in range(8)])

    def batch_compare(self, groups: list[list], points: list[int], target: str) -> list[Elimin8Comparison]:
        prompts = []
        for group in groups:
            if len(group) != 8:
                raise ValueError(f"Group must have exactly 8 items, got {len(group)}")

            context = {"target": target}
            for i, item in enumerate(group):
                context[f"item_{i+1}_id"] = item.obj_id
                context[f"item_{i+1}_info"] = item.obj_info
            prompts.append(self.template.format(**context))

        responses = self.session.batch_generate(prompts, response_model=Elimin8Response)

        results = []
        for group, response in zip(groups, responses.responses):
            if not response.message.structured_output:
                raise ValueError(f"Failed to parse structured output for group: {[item.obj_id for item in group]}. Response: {response.content[:200]}")
            
            ranked = response.message.structured_output.to_list()

            for rank_idx, ranked_item in enumerate(ranked):
                item = next((x for x in group if x.obj_id == ranked_item), None)
                if item:
                    item.score += points[rank_idx]

            results.append(Elimin8Comparison(
                group_items=[item.obj_id for item in group],
                rankings=ranked,
            ))

        return results


class RoundRobinComparator(BaseComparator):
    def __init__(self, session: Session, template: str):
        super().__init__(session, template, ["target", "item_1_id", "item_1_info", "item_2_id", "item_2_info"])
    
    def batch_compare(self, pairs: list[tuple], target: str) -> list:
        prompts = []
        for item1, item2 in pairs:
            prompts.append(self.template.format(
                target=target,
                item_1_id=item1.obj_id,
                item_1_info=item1.obj_info,
                item_2_id=item2.obj_id,
                item_2_info=item2.obj_info
            ))
        
        responses = self.session.batch_generate(prompts, response_model=RoundRobinMatchResult)
        
        winners = []
        for (item1, item2), response in zip(pairs, responses.responses):
            if not response.structured_output:
                raise ValueError(f"Failed to parse structured output for pair: {item1.obj_id} vs {item2.obj_id}. Response: {response.content[:200]}")
            result = response.structured_output
            winner_id = result.winner_id
            
            if winner_id not in [item1.obj_id, item2.obj_id]:
                raise ValueError(f"Invalid winner: {winner_id} not in [{item1.obj_id}, {item2.obj_id}]")
            
            winner = item1 if winner_id == item1.obj_id else item2
            loser = item2 if winner == item1 else item1
            
            winner.wins += 1
            winner.score_diff += result.score_difference
            loser.score_diff -= result.score_difference
            
            self.comparison_history.append(RoundRobinComparison(
                item_1_id=item1.obj_id,
                item_2_id=item2.obj_id,
                winner_id=winner.obj_id,
                score_difference=result.score_difference,
            ))
            
            winners.append(winner)
        
        return winners


# Sorters
class Elimin8Sorter:
    def __init__(self, items: list[Elimin8Item], comparator: Elimin8Comparator, target: str, points: list[int] = [10, 7, 5, 3, 2, 1, 0, -1], rounds_before_elimination: int = 1):
        self.items = items
        self.comparator = comparator
        self.target = target
        self.points = points
        self.rounds_before_elimination = rounds_before_elimination

        if len(self.points) != 8:
            raise ValueError("Points must have exactly 8 values")
        
        self.elimination_history = []
        self.round_history = []
    
    def _create_groups(self, items: list[Elimin8Item]) -> tuple[list[list[Elimin8Item]], int]:
        if not items:
            return [], 0

        sorted_items = sorted(items, key=lambda x: (-x.score, x.rank))

        group_count = math.ceil(len(sorted_items) / 8)
        filler_count = group_count * 8 - len(sorted_items)

        sorted_items = sorted_items + [Elimin8Item(obj_id=f"__FILLER_{i}__", obj_info="This item is just for filler purposes, it should be the last element", rank=-1) for i in range(filler_count)]

        buckets = []
        start = 0
        for i in range(8):
            size = len(sorted_items) // 8
            buckets.append(sorted_items[start:start + size])
            start += size
        
        for bucket in buckets:
            random.shuffle(bucket)

        groups = [[b[i] for b in buckets] for i in range(group_count)]
        return groups, filler_count

    def extract_top_k(self, k: int) -> list[tuple[str, str]]:
        active = copy.deepcopy(self.items)
        
        self.round_history.append([item.obj_id for item in active if not item.obj_id.startswith("__FILLER_")])
        
        while len(active) > k:
            for _ in range(self.rounds_before_elimination):
                groups, filler_count = self._create_groups(active)
                results = self.comparator.batch_compare(groups, self.points, self.target)
                self.elimination_history.extend(results)

            sorted_items = sorted(active, key=lambda x: x.score, reverse=True)
            target_size = max((len(sorted_items) + filler_count) // 2, k)
            active = sorted_items[:target_size]
            
            self.round_history.append([item.obj_id for item in active if not item.obj_id.startswith("__FILLER_")])

        finalists = sorted(active, key=lambda x: x.score, reverse=True)
        return [(item.obj_id, item.obj_info) for item in finalists]
    
    @staticmethod
    def topk(ranking_input: RankingInput, k: int, comparator: Elimin8Comparator, target: str, points: list[int] = None, rounds_before_elimination: int = 1) -> RankingSession:
        if k <= 0 or k > len(ranking_input.items):
            raise ValueError(f"k must be between 1 and {len(ranking_input.items)}")
        
        items = [Elimin8Item(item.id, item.description, rank=i) for i, item in enumerate(ranking_input.items)]
        
        sorter = Elimin8Sorter(items, comparator, target, points, rounds_before_elimination)
        ranked_tuples = sorter.extract_top_k(k)
        
        return RankingSession.from_tuples(ranked_tuples=ranked_tuples, history=sorter.elimination_history, round_history=sorter.round_history)


class RoundRobinSorter:
    def __init__(self, items: list[RoundRobinItem], comparator: RoundRobinComparator, target: str):
        self.items = items
        self.comparator = comparator
        self.target = target
    
    def run(self) -> list[tuple[str, str]]:        
        pairs = []
        for i in range(len(self.items)):
            for j in range(i + 1, len(self.items)):
                pairs.append((self.items[i], self.items[j]))

        self.comparator.batch_compare(pairs, self.target)
        
        return sorted(self.items, key=lambda x: (x.wins, x.score_diff), reverse=True)
    
    @staticmethod
    def rank(ranking_input: RankingInput, comparator: RoundRobinComparator, target: str) -> RankingSession:
        if not ranking_input.items:
            raise ValueError("Cannot rank empty list")
        
        items = [RoundRobinItem(item.id, item.description, rank=i, wins=0, score_diff=0) for i, item in enumerate(ranking_input.items)]

        sorter = RoundRobinSorter(items, comparator, target)
        ranked_tuples = [(item.obj_id, item.obj_info) for item in sorter.run()]
        
        return RankingSession.from_tuples(ranked_tuples=ranked_tuples, history=comparator.comparison_history)

