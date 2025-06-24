from abc import ABC, abstractmethod
from typing import Any, override

from pydantic import BaseModel

from .template_registry import register_template
from ...utils.data_utils import CaseDataDict
from ...utils.tools import enable_bracket_access


@enable_bracket_access
class ChargeFormatteredSample(BaseModel):
    fact: str
    charge_name: str


@enable_bracket_access
class MultiChargeFormatteredSample(BaseModel):
    fact: str
    charge_list: list[str]


@enable_bracket_access
class ImprisonmentFormatteredSample(BaseModel):
    fact: str
    imprisonment: int


class BaseFormatter(ABC):
    @abstractmethod
    def check_validation(self, raw_sample: dict[str, Any]) -> bool:
        pass

    def format_charge_sample(
        self, raw_sample: dict[str, Any]
    ) -> list[ChargeFormatteredSample]:
        return [
            ChargeFormatteredSample(fact=sample.fact, charge_name=sample.charge_list[0])
            for sample in self.format_multi_charge_sample(raw_sample)
        ]

    @abstractmethod
    def format_multi_charge_sample(
        self, raw_sample: dict[str, Any]
    ) -> list[MultiChargeFormatteredSample]:
        pass

    @abstractmethod
    def format_imprisonment_sample(
        self, raw_sample: dict[str, Any]
    ) -> list[ImprisonmentFormatteredSample]:
        pass


@register_template("Course")
class CourseFormatter(BaseFormatter):
    @override
    def check_validation(self, raw_sample: dict[str, Any]) -> tuple[bool, int, int]:
        charge_cnt = 0
        imprisonment_cnt = 0
        try:
            case_data_dict = CaseDataDict(**raw_sample)
            defendants = case_data_dict.defendants
            outcomes = case_data_dict.outcomes
            for defendant in defendants:
                is_in_outcome = False
                charge_cnt += 1
                for outcome in outcomes:
                    if defendant == outcome.name:
                        is_in_outcome = True
                        break
                if not is_in_outcome:
                    return False, 0, 0
            for outcome in outcomes:
                imprisonment_cnt += len(outcome.judgment)
                if not outcome.name in defendants:
                    return False, 0, 0
        except:
            return False, 0, 0
        return True, charge_cnt, imprisonment_cnt

    @override
    def format_multi_charge_sample(
        self, raw_sample: dict[str, Any]
    ) -> list[MultiChargeFormatteredSample]:
        case_data_dict = CaseDataDict(**raw_sample)
        outcome_list = case_data_dict.outcomes
        result_list = []
        for outcome in outcome_list:
            result_dict = {}
            result_dict["fact"] = (
                f"【当前被告人：{outcome.name}，" + case_data_dict.fact
            )
            result_dict["charge_list"] = outcome.standard_accusation
            result_list.append(MultiChargeFormatteredSample(**result_dict))
        return result_list

    @override
    def format_imprisonment_sample(
        self, raw_sample: dict[str, Any]
    ) -> list[ImprisonmentFormatteredSample]:
        case_data_dict = CaseDataDict(**raw_sample)
        outcome_list = case_data_dict.outcomes
        result_list = []
        for outcome in outcome_list:
            for judgment in outcome.judgment:
                result_dict = {}
                result_dict["fact"] = (
                    f"【当前被告人：{outcome.name}，【罪名：{judgment.standard_accusation}"
                    + case_data_dict.fact
                )
                result_dict["imprisonment"] = judgment.imprisonment
                result_list.append(ImprisonmentFormatteredSample(**result_dict))
        return result_list
