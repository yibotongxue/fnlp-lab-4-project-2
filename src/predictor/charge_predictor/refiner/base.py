from abc import ABC, abstractmethod


class BaseRefiner(ABC):
    def __init__(self, candidate_cnt: int):
        self.candidate_cnt = candidate_cnt

    @abstractmethod
    def refine(self, fact: str, defendant: str, candidate: list[str]) -> str:
        pass

    def refine_batch(
        self, fact: str, defendants: list[str], candidate: list[list[str]]
    ) -> list[str]:
        return [
            self.refine(fact, defendant, candidate)
            for defendant, candidate in zip(defendants, candidate)
        ]
