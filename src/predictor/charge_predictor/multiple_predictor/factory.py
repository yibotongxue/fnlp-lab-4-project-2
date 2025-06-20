from argparse import Namespace

from .base import BaseMultipleChargePredictor
from .lawformer import LawformerMultipleChargePredictor
from ....utils.data_utils import ChargeLoader


def get_multiple_charge_predictor(
    predictor_type: str, args: Namespace
) -> BaseMultipleChargePredictor:
    """
    Factory function to get the appropriate charge predictor based on the type.

    Args:
        predictor_type (str): Type of the charge predictor.
        args (Namespace): Arguments containing model paths and configurations.

    Returns:
        BaseChargePredictor: An instance of the specified charge predictor.
    """
    if predictor_type == "lawformer":
        charge_loader = ChargeLoader(args.charge_file)
        return LawformerMultipleChargePredictor(
            args.candidate_cnt,
            args.multiple_charge_model_path,
            args.multiple_charge_base_model_name,
            charge_num=charge_loader.charge_num,
            charge_id_mapping=charge_loader.reverse_charges,
            device=args.multiple_device,
        )
    else:
        raise ValueError(f"Unknown charge predictor type: {predictor_type}")
