from typing import List

from pydantic import BaseModel


class JudgmentDict(BaseModel):
    standard_accusation: str
    imprisonment: int


class OutcomeDict(BaseModel):
    name: str
    judgment: List[JudgmentDict]


class CaseDataDict(BaseModel):
    fact: str
    defendants: List[str]
    outcomes: List[OutcomeDict]
