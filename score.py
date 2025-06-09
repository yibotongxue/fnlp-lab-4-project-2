import argparse
import os

import pandas as pd

from src.metrics.metric_charge import score as charge_score
from src.metrics.metric_imprisonment import score as imprisonment_score
from src.utils import load_json
from src.utils.type_utils import OutcomeDict

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
    solution_data = {"id": [], "gold_accusation": [], "gold_imprisonment": []}
    submission_data = {"id": [], "accusations": [], "imprisonment": []}
    for json_file in os.listdir(os.path.join(output_dir, "zero_shot")):
        json_file = os.path.join(output_dir, "zero_shot", json_file)
        data = load_json(json_file)
        solution_outcome = data["input"]["outcomes"]
        submission_outcome = data["result"]
        if not len(solution_outcome) == len(submission_outcome):
            print(
                f"Skipping {json_file} due to mismatched outcome lengths. Len of solution: {len(solution_outcome)}, len of submission: {len(submission_outcome)}"
            )
            continue
        try:
            solution_outcome = [OutcomeDict(**outcome) for outcome in solution_outcome]
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        try:
            submission_outcome = [
                OutcomeDict(**outcome) for outcome in submission_outcome
            ]
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        solution_data["id"].append(json_file.split("/")[-1].split(".")[0])
        submission_data["id"].append(json_file.split("/")[-1].split(".")[0])
        solution_data["gold_accusation"].append(
            ";".join([outcome.get_accusation_str() for outcome in solution_outcome])
        )
        solution_data["gold_imprisonment"].append(
            str([outcome.imprisonment for outcome in solution_outcome])
        )
        submission_data["accusations"].append(
            ";".join([outcome.get_accusation_str() for outcome in submission_outcome])
        )
        submission_data["imprisonment"].append(
            str([outcome.imprisonment for outcome in submission_outcome])
        )

    solution = pd.DataFrame(solution_data)
    submission = pd.DataFrame(submission_data)
    row_id_column_name = "id"

    charge_score_result = charge_score(solution, submission, row_id_column_name)
    print(f"Charge Case-level F1 score: {charge_score_result}")
    imprisonment_score_result = imprisonment_score(
        solution, submission, row_id_column_name
    )
    print(f"Imprisonment Case-level F1 score: {imprisonment_score_result}")
