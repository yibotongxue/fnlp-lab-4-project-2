import os

import pandas as pd

from src.metrics.metric_charge import score as charge_score
from src.metrics.metric_imprisonment import score as imprisonment_score
from src.utils import load_json
from src.utils.type_utils import OutcomeDict

if __name__ == "__main__":
    solution_data = {"id": [], "gold_accusation": [], "gold_imprisonment": []}
    submission_data = {"id": [], "accusations": [], "imprisonment": []}
    for i in range(2):
        json_file = os.path.join("output", "zero_shot", f"result_{i}.json")
        solution_data["id"].append(str(i + 1))
        submission_data["id"].append(str(i + 1))
        data = load_json(json_file)
        solution_outcome = data["input"]["outcomes"]
        solution_outcome = [OutcomeDict(**outcome) for outcome in solution_outcome]
        submission_outcome = data["result"]
        submission_outcome = [OutcomeDict(**outcome) for outcome in submission_outcome]
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
