from argparse import Namespace

from .base import BaseRefiner
from .zero_shot import ZeroShotRefiner


def get_refiner(refiner_type: str, args: Namespace) -> BaseRefiner:
    if refiner_type == "zero_shot":
        return ZeroShotRefiner(args.candidate_cnt, args.refiner_llm)
    else:
        raise ValueError(f"Unsupported refiner type {refiner_type}")
