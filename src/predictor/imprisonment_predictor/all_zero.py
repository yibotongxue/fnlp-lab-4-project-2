from typing import override

from .base import BaseImprisonmentPredictor


class AllZeroImprisonmentPredictor(BaseImprisonmentPredictor):
    @override
    def predict_imprisonment(
        self, fact: str, defendants: list[str], charge_dict: dict[str, list[str]]
    ) -> dict:
        # Implement the logic to predict imprisonment using Lawformer
        result = {}
        for defendant in defendants:
            result[defendant] = []
            for charge in charge_dict.get(defendant, []):
                result[defendant].append(
                    {"standard_accusation": charge, "imprisonment": 0}
                )
        return result
