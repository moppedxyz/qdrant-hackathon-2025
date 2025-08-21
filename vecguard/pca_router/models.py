from typing import List, NamedTuple


class Route(NamedTuple):
    name: str
    utterances: List[str]


class RouterPrediction(NamedTuple):
    route: str
    similarity_score: float
