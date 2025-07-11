import os
import argparse
from src.utils import load_jsonl, save_jsonl
from src.utils.data_utils import ChargeLoader


def flatten_cases(
    raw_cases: list[dict], charge_id_map: dict[str, int], is_train: bool = True
) -> list[dict]:
    flat_samples = []

    for case in raw_cases:
        fact = case["fact"]
        name = case["meta"]["criminals"][0]
        charge = case["meta"]["accusation"][0].replace("[", "").replace("]", "")
        # print(charge)
        imprisonment = case["meta"]["term_of_imprisonment"]["imprisonment"]

        if not is_train:
            sample = {
                "fact": f"【当前被告人：{name}】" + fact,
                "defendant": name,
            }
            flat_samples.append(sample)
            continue
        if not charge in charge_id_map:
            continue
        sample = {
            "fact": f"【当前被告人：{name}】" + fact,
            "fact_imprisonment": f"【当前被告人：{name}】，【罪名：{charge}】" + fact,
            "defendant": name,
            "charge": charge,
            "charge_id": charge_id_map[charge],
            "imprisonment": imprisonment,
        }
        flat_samples.append(sample)

    return flat_samples


def flatten_jsonl_file(
    input_path: str,
    charge_file: str,
    output_path: str,
    overwrite: bool = False,
    is_train: bool = True,
):
    if os.path.exists(output_path) and not overwrite:
        print(f"[INFO] Skipped: {output_path} already exists.")
        return

    charge_loader = ChargeLoader(charge_file)

    raw_cases = load_jsonl(input_path)
    flat_samples = flatten_cases(raw_cases, charge_loader.all_charges, is_train)
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
    parser.add_argument(
        "--is-train",
        action="store_true",
        help="Whether the dataset is for training (default: False)",
    )

    args = parser.parse_args()
    flatten_jsonl_file(
        args.input, args.charge, args.output, args.overwrite, args.is_train
    )
