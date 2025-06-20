from typing import override

from .base import BaseChargePredictor
from .refiner import BaseRefiner
from .multiple_predictor import BaseMultipleChargePredictor


class RefineChargePredictor(BaseChargePredictor):
    def __init__(
        self, refiner: BaseRefiner, multiple_predictor: BaseMultipleChargePredictor
    ):
        self.refiner = refiner
        self.multiple_predictor = multiple_predictor

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        candidates = self.multiple_predictor.predict(fact, defendants)
        result = {}
        refined_charge = self.refiner.refine(
            fact, defendants[0], candidates[defendants[0]]
        )
        for defendant in candidates.keys():
            result[defendant] = refined_charge
        return result
