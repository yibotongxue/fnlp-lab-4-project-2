from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        pass

    def retrieve_batch(self, queries: list[str], k: int = 3) -> list[list[str]]:
        return [self.retrieve(query, k) for query in queries]
