from ...utils.imprisonment_mapper import BaseImprisonmentMapper

import torch


class CustomDataCollator:
    def __init__(
        self, imprisonment_mapper: BaseImprisonmentMapper, is_charge: bool = True
    ):
        self.imprisonment_mapper = imprisonment_mapper
        self.is_charge = is_charge

    def __call__(self, features: list[dict]) -> dict:
        """
        Custom data collator for processing features into a batch.

        Args:
            features (list[dict]): List of feature dictionaries.

        Returns:
            dict: A dictionary containing batched input_ids, attention_mask, and labels if available.
        """

        # Stack input_ids and attention_mask
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])

        # 如果是训练集，包含标签
        if "label" in features[0]:
            charge_ids = torch.stack([f["label"]["charge_id"] for f in features])
            imprisonments = torch.tensor(
                [
                    self.imprisonment_mapper.imprisonment2label(
                        f["label"]["imprisonment"]
                    )
                    for f in features
                ],
                dtype=torch.long,
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": charge_ids if self.is_charge else imprisonments,
            }
        else:
            # 推理时无标签
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
