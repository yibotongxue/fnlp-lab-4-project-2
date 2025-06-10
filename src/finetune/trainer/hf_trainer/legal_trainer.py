from transformers import Trainer
from torch.nn import functional as F

from ...model import LegalPredictionOutput, LegalPredictionModel


class LegalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        Initializes the LegalTrainer with the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.charge_weight = kwargs.pop("charge_weight", 1.0)
        self.imprisonment_weight = kwargs.pop("imprisonment_weight", 1.0)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model: LegalPredictionModel, inputs: dict, return_outputs=False
    ):
        """
        Computes the loss for the model given the inputs.

        Args:
            model: The model to compute the loss for.
            inputs: The inputs to the model.
            return_outputs (bool): Whether to return the outputs of the model.

        Returns:
            The computed loss and optionally the outputs of the model.
        """
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        outputs: LegalPredictionOutput = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        charge_logits = outputs.charge_logits
        imprisonment_logits = outputs.imprisonment_logits
        charge_labels = inputs["labels"]["charge_id"]
        imprisonment_labels = inputs["labels"]["imprisonment"]
        charge_loss = self._compute_charge_loss(charge_logits, charge_labels)
        imprisonment_loss = self._compute_imprisonment_loss(
            imprisonment_logits, imprisonment_labels, charge_labels
        )
        total_loss = (
            charge_loss * self.charge_weight
            + imprisonment_loss * self.imprisonment_weight
        )
        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _compute_charge_loss(self, charge_logits, charge_labels):
        """
        Computes the charge loss using cross-entropy loss.

        Args:
            charge_logits: The logits for the charge predictions.
            charge_labels: The labels for the charge predictions.

        Returns:
            The computed charge loss.
        """
        return F.cross_entropy(
            charge_logits,
            charge_labels,
            ignore_index=-1,  # Ignore the -1 label
            reduction="mean",  # Average loss over the batch
        )

    def _compute_imprisonment_loss(
        self, imprisonment_logits, imprisonment_labels, charge_labels
    ):
        """
        Computes the imprisonment loss using cross-entropy loss.
        Args:
            imprisonment_logits: The logits for the imprisonment predictions.
            imprisonment_labels: The labels for the imprisonment predictions.
            charge_labels: The labels for the charge predictions.
        Returns:
            The computed imprisonment loss.
        """

        # Filter out imprisonment labels where charge label is -1
        valid_mask = charge_labels != -1
        valid_imprisonment_logits = imprisonment_logits[valid_mask]
        valid_imprisonment_labels = imprisonment_labels[valid_mask]

        return F.cross_entropy(
            valid_imprisonment_logits,
            valid_imprisonment_labels,
            ignore_index=-1,  # Ignore the -1 label
            reduction="mean",  # Average loss over the batch
        )
