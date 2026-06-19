import torch
import torch.nn as nn

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module implementing the cross-entropy loss function for sequence-to-sequence models.
    """

    def __init__(self, num_classes: int, padding_idx: int = 0, smoothing: float = 0.1):
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - padding_idx (int): The index of the padding token. Defaults to 0.
        - smoothing (float): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=padding_idx)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The cross-entropy loss.
        """
        return self.criterion(predictions, targets)

if __name__ == "__main__":
    # Create a dummy dataset
    num_classes = 10
    batch_size = 32
    sequence_length = 20
    padding_idx = 0

    # Initialize the loss function
    loss_fn = CrossEntropyLossFunction(num_classes, padding_idx)

    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, sequence_length, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, sequence_length))

    # Compute the loss
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item():.4f}")