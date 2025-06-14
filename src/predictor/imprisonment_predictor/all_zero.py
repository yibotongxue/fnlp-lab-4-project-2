from typing import override

from .base import BaseImprisonmentPredictor


class AllZeroImprisonmentPredictor(BaseImprisonmentPredictor):
    @override
    def predict(
        self, fact: str, defendants: list[str], charge_dict: dict[str, list[str]]
    ) -> dict:
        result = {}
        for defendant in defendants:
            result[defendant] = []
            for charge in charge_dict.get(defendant, []):
                result[defendant].append(
                    {"standard_accusation": charge, "imprisonment": 0}
                )
        return result
