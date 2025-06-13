from argparse import Namespace

from .base import BaseChargePredictor
from .zero_shot import ZeroShotChargePredictor
from .lawformer import LawformerChargePredictor
from ...utils.data_utils import ChargeLoader


def get_charge_predictor(predictor_type: str, args: Namespace) -> BaseChargePredictor:
    """
    Factory function to get the appropriate charge predictor based on the type.

    Args:
        predictor_type (str): Type of the charge predictor.
        args (Namespace): Arguments containing model paths and configurations.

    Returns:
        BaseChargePredictor: An instance of the specified charge predictor.
    """
    if predictor_type == "zero_shot":
        return ZeroShotChargePredictor(args.charge_predictor, args.charge_llm)
    elif predictor_type == "lawformer":
        charge_loader = ChargeLoader(args.charge_file)
        return LawformerChargePredictor(
            args.charge_model_path,
            args.charge_base_model_name,
            charge_num=charge_loader.charge_num,
            charge_id_mapping=charge_loader.reverse_charges,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown charge predictor type: {predictor_type}")
