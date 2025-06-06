from typing import List

from pydantic import BaseModel


class JudgmentDict(BaseModel):
    standard_accusation: str
    imprisonment: int

    def __getitem__(self, item):
        if item == "standard_accusation":
            return self.standard_accusation
        elif item == "imprisonment":
            return self.imprisonment
        else:
            raise KeyError(f"Invalid key: {item}")


class OutcomeDict(BaseModel):
    name: str
    judgment: List[JudgmentDict]

    def __getitem__(self, item):
        if item == "name":
            return self.name
        elif item == "judgment":
            return self.judgment
        else:
            raise KeyError(f"Invalid key: {item}")


class CaseDataDict(BaseModel):
    fact: str
    defendants: List[str]
    outcomes: List[OutcomeDict]

    def __getitem__(self, item):
        if item == "fact":
            return self.fact
        elif item == "defendants":
            return self.defendants
        elif item == "outcomes":
            return self.outcomes
        else:
            raise KeyError(f"Invalid key: {item}")
