from abc import ABC, abstractmethod
from typing import override


class BaseImprisonmentMapper(ABC):
    @abstractmethod
    def imprisonment2label(self, imprisonment: int) -> int:
        pass

    @abstractmethod
    def label2imprisonment(self, label: int) -> int:
        pass


class IdentityImprisonmentMapper(BaseImprisonmentMapper):
    @override
    def imprisonment2label(self, imprisonment: int) -> int:
        return imprisonment

    @override
    def label2imprisonment(self, label: int) -> int:
        return label


class IntervalImprisonmentMapper(BaseImprisonmentMapper):
    def __init__(self, lower_bound: list[int], represent_list: list[int]):
        assert len(lower_bound) >= 1, "lower bounds should be more than one element"
        assert (
            lower_bound[0] == 0
        ), f"the first lower bound should be zero, but is {lower_bound[0]} now"
        assert len(lower_bound) == len(
            represent_list
        ), f"the length of lower bound and represent list should be the same, but the length of lower bound is {len(lower_bound)} and the length of the represent list is {len(represent_list)}"
        for i in range(len(lower_bound) - 1):
            assert (
                lower_bound[i] < lower_bound[i + 1]
            ), f"the lower bound should be incresement, now is {lower_bound}"
        for i in range(len(represent_list) - 1):
            assert (
                represent_list[i] < represent_list[i + 1]
            ), f"the represent list should be incresement, now is {represent_list}"
        self.lower_bound = lower_bound
        self.represent_list = represent_list

    @override
    def imprisonment2label(self, imprisonment: int) -> int:
        assert (
            imprisonment >= 0
        ), f"imprisonment should be not less than zero, but is {imprisonment} now"
        for i in range(len(self.lower_bound) - 1, -1, -1):
            if imprisonment >= self.lower_bound[i]:
                return i
        raise ValueError(f"not found imprisonment interval")

    @override
    def label2imprisonment(self, label: int) -> int:
        assert label >= 0 and label < len(
            self.lower_bound
        ), f"label should be not less than zero and less than length of represent list, which length is {len(self.represent_list)}, but now the label is {label}"
        return self.represent_list[label]


def get_imprisonment_mapper(imprisonment_mapper_config: dict) -> BaseImprisonmentMapper:
    if imprisonment_mapper_config["imprisonment_mapper_type"] == "identity":
        return IdentityImprisonmentMapper()
    elif imprisonment_mapper_config["imprisonment_mapper_type"] == "interval":
        return IntervalImprisonmentMapper(
            lower_bound=imprisonment_mapper_config["lower_bound"],
            represent_list=imprisonment_mapper_config["represent_list"],
        )
    else:
        raise ValueError(
            f"Unsupported imprisonment mapper type {imprisonment_mapper_config['imprisonment_mapper_type']}"
        )
