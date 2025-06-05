import json
import math

import pandas as pd
import pandas.api.types
import torch


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def get_weight(defendant_num):
    """
        给定案件被告人人数，计算基础权重
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        float: 权重值，计算方式为log2(D), D为被告人数量
    """
    if isinstance(defendant_num, torch.Tensor):
        # defendant_num为tensor类型
        return math.log(defendant_num.item(), 2)
    else:
        # defendant_num为直接的数值类型
        return math.log(defendant_num, 2)


def get_per_defendant_metrics_weight(defendant_num):
    """
        计算最终结果中每个被告人对应的权重，是get_weight(defendant_num)/defendant_num
    Args:
        defendant_num (_type_): 案件被告人数量
    Returns:
        _type_: _description_
    """
    if isinstance(defendant_num, torch.Tensor):
        assert defendant_num.item() != 0
        return get_weight(defendant_num.item()) / defendant_num.item()
    else:
        assert defendant_num != 0
        return get_weight(defendant_num) / defendant_num


def get_weighted_per_defendant_metrics(outputs, labels, weight_tensor):
    """
        通过计算per-defendant指标计算per-case指标后进行加权
    Args:
        outputs (_type_): 模型预测
        labels (_type_): 真实标签
        weight_tensor (_type_): 这里的每个weight为get_weight(defendant_num)/defendant_num
    Returns:
        acc, precision, recall, f1: 加权后的最终指标
    """
    assert outputs.shape == labels.shape
    assert outputs.size(0) == weight_tensor.size(0)
    acc, precision, recall, f1 = 0, 0, 0, 0
    outputs.size(0)
    # 加权平均的分母
    weight_total = torch.sum(weight_tensor).item()
    TPs = torch.sum(outputs * labels, dim=1)
    FPs = torch.sum(outputs * (1 - labels), dim=1)
    FNs = torch.sum((1 - outputs) * labels, dim=1)
    TNs = torch.sum((1 - outputs) * (1 - labels), dim=1)
    precisions = TPs / torch.clamp(TPs + FPs, min=1e-8)
    recalls = TPs / torch.clamp(TPs + FNs, min=1e-8)
    f1s = 2 * precisions * recalls / torch.clamp(precisions + recalls, min=1e-8)
    # exact match accuracy
    acc = (
        torch.sum(weight_tensor * torch.all(outputs == labels, dim=1).float())
        / weight_total
    )
    precision = torch.sum(weight_tensor * precisions) / weight_total
    recall = torch.sum(weight_tensor * recalls) / weight_total
    f1 = torch.sum(weight_tensor * f1s) / weight_total
    return acc.item(), precision.item(), recall.item(), f1.item()


def get_case_level_metrics_by_per_defendant_metrics(outputs, labels, defendant_nums):
    """
        计算case-level metrics
    Args:
        outputs (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        labels (torch.Tensor): size(batch_size, num_labels)，每个元素为0或1
        defendant_nums (List): len(batch_size) 每条输入对应案件的被告人数量数组
    Returns:
        三个任务的结果
    """
    # 对每个被告人计算accuracy, precision, recall, f1
    weight_tensor = torch.tensor(
        list(map(get_per_defendant_metrics_weight, defendant_nums))
    ).to(outputs.device)
    charge_acc, charge_precision, charge_recall, charge_f1 = (
        get_weighted_per_defendant_metrics(outputs, labels, weight_tensor)
    )
    result = {
        "case_level_Acc_Acc": charge_acc,
        "case_level_Acc_Precision": charge_precision,
        "case_level_Acc_Recall": charge_recall,
        "case_level_Acc_F1": charge_f1,
    }
    # print(result)
    return result


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    计算案件级别的F1分数，基于被告人级别的预测结果加权平均

    该函数将solution和submission中的刑期预测结果转换为one-hot编码，
    然后计算加权平均的case-level指标，最终返回F1分数。

    Args:
        solution (pd.DataFrame): 包含真实标签的数据框，必须有'id'和'gold_imprisonment'列
        submission (pd.DataFrame): 包含预测结果的数据框，必须有'id'和'imprisonment'列
        row_id_column_name (str): 用于匹配solution和submission的ID列名

    Returns:
        float: 加权平均后的case-level F1分数

    Note:
        - 输入样例参考sample_submission.csv
        - 权重计算基于被告人数量，使用log2(D)/D的公式
        - 最终分数是case-level的加权平均F1分数
    """

    # 转换为one-hot编码
    flat_predictions = []
    flat_labels = []
    defendant_counts = []

    for _, row in submission.iterrows():
        case_id = row["id"]
        solution_row = solution[solution["id"] == case_id].iloc[0]

        # 获取每个被告人的罪名预测和真实标签
        pred_imps = json.loads(row["imprisonment"])
        gold_imps = json.loads(solution_row["gold_imprisonment"])

        num_defendants = len(pred_imps)
        defendant_counts.extend([num_defendants] * num_defendants)

        # 转换为one-hot
        for i in range(num_defendants):
            pred_one_hot = [0] * 600
            for imp in pred_imps[i]:
                pred_one_hot[int(imp)] = 1

            gold_one_hot = [0] * 600
            for imp in gold_imps[i]:
                if imp in pred_imps[i]:
                    gold_one_hot[int(imp)] = 1

            flat_predictions.append(pred_one_hot)
            flat_labels.append(gold_one_hot)

    # 转换为tensor
    predictions_tensor = torch.tensor(flat_predictions, dtype=torch.float32)
    labels_tensor = torch.tensor(flat_labels, dtype=torch.float32)
    defendant_nums = torch.tensor(defendant_counts, dtype=torch.long)

    # 计算指标
    result = get_case_level_metrics_by_per_defendant_metrics(
        predictions_tensor, labels_tensor, defendant_nums
    )

    # 返回F1分数作为最终得分
    return result["case_level_Acc_F1"]


if __name__ == "__main__":
    solution_data = {
        "id": ["1", "2", "3"],
        "gold_imprisonment": ["[[9], [9]]", "[[12], [10]]", "[[7], [9], [6]]"],
    }
    submission_data = {
        "id": ["1", "2", "3"],
        "imprisonment": ["[[9], [9,20]]", "[[9], [10]]", "[[7], [9], [6]]"],
    }
    solution = pd.DataFrame(solution_data)
    submission = pd.DataFrame(submission_data)
    row_id_column_name = "id"
    print(score(solution, submission, row_id_column_name))
