import os
import argparse
from src.utils import load_jsonl, save_jsonl
from src.utils.type_utils import CaseDataDict
from src.utils.data_utils import ChargeLoader


def flatten_cases(raw_cases: list[dict], charge_id_map: dict[str, int]) -> list[dict]:
    flat_samples = []

    for case in raw_cases:
        case = CaseDataDict(**case)
        fact = case["fact"]
        defendants = case["defendants"]
        outcomes = case["outcomes"]

        outcome_map = {o["name"]: o["judgment"][0] for o in outcomes}

        for name in defendants:
            if name not in outcome_map:
                continue
            judgment = outcome_map[name]
            assert (
                judgment["standard_accusation"] in charge_id_map
            ), f"Standard accusation '{judgment['standard_accusation']}' not found in charge_id_map."
            sample = {
                "fact": f"【当前被告人：{name}】" + fact,
                "defendant": name,
                "charge": judgment["standard_accusation"],
                "charge_id": charge_id_map[judgment["standard_accusation"]],
                "imprisonment": judgment["imprisonment"],
            }
            flat_samples.append(sample)

    return flat_samples


def flatten_jsonl_file(
    input_path: str, charge_file: str, output_path: str, overwrite: bool = False
):
    if os.path.exists(output_path) and not overwrite:
        print(f"[INFO] Skipped: {output_path} already exists.")
        return

    charge_loader = ChargeLoader(charge_file)

    raw_cases = load_jsonl(input_path)
    flat_samples = flatten_cases(raw_cases, charge_loader.all_charges)
    save_jsonl(flat_samples, output_path)
    print(f"[INFO] Flattened {len(flat_samples)} samples to {output_path}")


# --- CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten multi-defendant cases to one-per-defendant samples."
    )
    parser.add_argument(
        "--input", type=str, help="Path to original JSONL file (multi-defendant)"
    )
    parser.add_argument("--charge", type=str, help="Path to charge mapping JSON file")
    parser.add_argument(
        "--output", type=str, help="Path to save the flattened JSONL file"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if exists"
    )

    args = parser.parse_args()
    flatten_jsonl_file(args.input, args.charge, args.output, args.overwrite)
