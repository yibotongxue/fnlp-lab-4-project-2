from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput
from safetensors.torch import load_file


@dataclass
class LegalPredictionOutput(ModelOutput):
    charge_logits: torch.FloatTensor = None
    imprisonment_logits: torch.FloatTensor = None


class LegalPredictionModel(nn.Module):
    def __init__(
        self, base_model: nn.Module, charge_num: int = 321, imprisonment_num: int = 600
    ):
        """
        Initializes the LegalPredictionModel with a base model.

        Args:
            base_model (nn.Module): The base model to be used for legal predictions.
        """
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.charge_classifier = nn.Linear(base_model.config.hidden_size, charge_num)
        self.config = base_model.config
        self.imprisonment_classifier = nn.Linear(
            base_model.config.hidden_size + charge_num, imprisonment_num
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> LegalPredictionOutput:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Model outputs.
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        charge_logits = self.charge_classifier(pooled_output)
        imprisonment_input = torch.cat((pooled_output, charge_logits), dim=-1)
        imprisonment_logits = self.imprisonment_classifier(imprisonment_input)
        return LegalPredictionOutput(
            charge_logits=charge_logits, imprisonment_logits=imprisonment_logits
        )

    @classmethod
    def from_pretrained(
        cls,
        safetensors_path: str,
        base_model_name: str,
        charge_num: int = 321,
        imprisonment_num: int = 600,
    ):
        """
        Load model weights from a SafeTensors file.

        Args:
            safetensors_path (str): Path to the SafeTensors file.
        """
        base_model = AutoModel.from_pretrained(base_model_name)
        state_dict = load_file(safetensors_path)
        model = cls(
            base_model, charge_num=charge_num, imprisonment_num=imprisonment_num
        )
        model.load_state_dict(state_dict, strict=False)
        return model
