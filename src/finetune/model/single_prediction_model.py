import torch
from torch import nn
from transformers import AutoModel
from safetensors.torch import load_file


class LegalSinglePredictionModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int = 321):
        """
        Initializes the LegalPredictionModel with a base model.

        Args:
            base_model (nn.Module): The base model to be used for legal predictions.
        """
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.charge_classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
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
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                charge_logits.view(-1, self.charge_classifier.out_features),
                labels.view(-1),
            )
            return loss, charge_logits
        return charge_logits

    @classmethod
    def from_pretrained(
        cls,
        safetensors_path: str,
        base_model_name: str,
        num_classes: int = 321,
    ):
        """
        Load model weights from a SafeTensors file.

        Args:
            safetensors_path (str): Path to the SafeTensors file.
        """
        base_model = AutoModel.from_pretrained(base_model_name)
        state_dict = load_file(safetensors_path)
        model = cls(
            base_model,
            num_classes=num_classes,
        )
        model.load_state_dict(state_dict, strict=False)
        return model
