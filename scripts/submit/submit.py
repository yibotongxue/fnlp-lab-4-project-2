import argparse
import os

import pandas as pd

from src.utils import load_json
from src.utils.data_utils import OutcomeDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score zero-shot predictions")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory containing the zero-shot prediction results",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    submission_task_1 = {"id": [], "accusations": []}
    submission_task_2 = {"id": [], "imprisonment": []}
    json_files = os.listdir(os.path.join(output_dir))
    json_files.sort(
        key=lambda x: (
            int(x.split(".")[0]) if x.split(".")[0].isdigit() else float("inf")
        )
    )
    for json_file in json_files:
        json_file = os.path.join(output_dir, json_file)
        if not json_file.endswith("json"):
            print(f"Skipping {json_file} as it is not a JSON file.")
            continue
        data = load_json(json_file)
        submission_outcome = data["result"]
        try:
            submission_outcome = [
                OutcomeDict(**outcome) for outcome in submission_outcome
            ]
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        submission_task_1["id"].append(int(json_file.split("/")[-1].split(".")[0]) + 1)
        submission_task_1["accusations"].append(
            ";".join([outcome.get_accusation_str() for outcome in submission_outcome])
        )
        submission_task_2["id"].append(int(json_file.split("/")[-1].split(".")[0]) + 1)
        submission_task_2["imprisonment"].append(
            str([outcome.imprisonment for outcome in submission_outcome])
        )

    submission_task_1 = pd.DataFrame(submission_task_1)
    submission_task_2 = pd.DataFrame(submission_task_2)
    # save the submission DataFrame to a CSV file
    submission_task_1.to_csv(
        os.path.join(output_dir, "./submission_task_1.csv"),
        index=False,
        encoding="utf-8",
    )
    submission_task_2.to_csv(
        os.path.join(output_dir, "./submission_task_2.csv"),
        index=False,
        encoding="utf-8",
    )
