from argparse import Namespace

from .base import BaseChargePredictor
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
        from .zero_shot import ZeroShotChargePredictor

        return ZeroShotChargePredictor(args.charge_predictor, args.charge_llm)
    elif predictor_type == "lawformer":
        from .lawformer import LawformerChargePredictor

        charge_loader = ChargeLoader(args.charge_file)
        return LawformerChargePredictor(
            args.charge_model_path,
            charge_num=charge_loader.charge_num,
            charge_id_mapping=charge_loader.reverse_charges,
            device=args.device,
        )
    elif predictor_type == "refine":
        from .refine import RefineChargePredictor
        from .refiner import get_refiner
        from .multiple_predictor import get_multiple_charge_predictor

        multiple_predictor = get_multiple_charge_predictor(
            args.multiple_predictor_type, args
        )
        refiner = get_refiner(args.refiner_type, args)
        return RefineChargePredictor(refiner, multiple_predictor)
    elif predictor_type == "multi-label":
        from .multi_label import MultiLabelChargePredictor

        charge_loader = ChargeLoader(args.charge_file)
        return MultiLabelChargePredictor(
            args.charge_model_path,
            charge_num=charge_loader.charge_num,
            charge_id_mapping=charge_loader.reverse_charges,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown charge predictor type: {predictor_type}")
