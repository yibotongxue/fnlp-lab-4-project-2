from typing import override

from .base import BaseChargePredictor


class VoterChargePredictor(BaseChargePredictor):
    def __init__(self, predictors: list[BaseChargePredictor]):
        """
        Initializes the VoterChargePredictor with a list of charge predictors.

        Args:
            predictors (list[BaseChargePredictor]): List of charge predictors to use.
        """
        self.predictors = predictors

    @override
    def predict(self, fact: str, defendants: list[str]) -> dict[str, list[str]]:
        """
        Predicts the charges for the given fact and defendants using multiple predictors.

        Args:
            fact (str): The fact to analyze.
            defendants (list[str]): List of defendants involved in the case.

        Returns:
            dict: A dictionary containing the predicted charges from each predictor.
        """
        results = {}
        for predictor in self.predictors:
            result = predictor.predict(fact, defendants)
            for defendant, charges in result.items():
                if defendant not in results:
                    results[defendant] = []
                results[defendant].extend(charges)
        for defendant in results:
            # vote and get the most common charge
            vote_count = {}
            for charge in results[defendant]:
                if charge not in vote_count:
                    vote_count[charge] = 0
                vote_count[charge] += 1
            # find the charge with the maximum votes
            max_charge = max(vote_count, key=vote_count.get)
            results[defendant] = [max_charge]
        return results
