from argparse import Namespace

from .base import BaseImprisonmentPredictor
from .zero_shot import ZeroShotImprisonmentPredictor
from .lawformer import LawformerImprisonmentPredictor
from .all_zero import AllZeroImprisonmentPredictor
from .most_common import MostCommonImprisonmentPredictor
from ...utils import load_config
from ...utils.imprisonment_mapper import get_imprisonment_mapper


def get_imprisonment_predictor(
    predictor_type: str, args: Namespace
) -> BaseImprisonmentPredictor:
    """
    Factory function to get the appropriate imprisonment predictor based on the type.

    Args:
        predictor_type (str): Type of the imprisonment predictor.
        args (Namespace): Arguments containing model paths and configurations.

    Returns:
        BaseImprisonmentPredictor: An instance of the specified imprisonment predictor.
    """
    if predictor_type == "zero_shot":
        return ZeroShotImprisonmentPredictor(args.imprisonment_llm)
    elif predictor_type == "lawformer":
        imprisonment_mapper = get_imprisonment_mapper(
            load_config(args.imprisonment_mapper_config)["imprisonment_mapper_config"]
        )
        return LawformerImprisonmentPredictor(
            args.imprisonment_model_path,
            imprisonment_num=args.imprisonment_num,
            device=args.device,
            imprisonment_mapper=imprisonment_mapper,
        )
    elif predictor_type == "all_zero":
        return AllZeroImprisonmentPredictor()
    elif predictor_type == "most_common":
        return MostCommonImprisonmentPredictor(args.charge_imprisonment_dict_path)
    else:
        raise ValueError(f"Unknown imprisonment predictor type: {predictor_type}")
