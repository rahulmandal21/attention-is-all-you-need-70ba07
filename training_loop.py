import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from typing import Callable, Tuple

class Trainer:
    """
    A class used to train a PyTorch model.

    Attributes:
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    optimizer : optim.Optimizer
        The optimizer used to update the model parameters.
    loss_fn : Callable
        The loss function used to calculate the loss.
    device : torch.device
        The device used to train the model.
    warmup_steps : int
        The number of warmup steps.
    d_model : int
        The dimension of the model.
    max_grad_norm : float
        The maximum gradient norm.

    Methods:
    -------
    train_one_epoch(dataloader: DataLoader) -> float
        Trains the model for one epoch.
    validate(dataloader: DataLoader) -> float
        Validates the model on a given dataloader.
    should_stop_early(val_loss: float) -> bool
        Checks if the training should stop early.
    save_checkpoint(path: str) -> None
        Saves the model checkpoint.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, loss_fn: Callable, device: torch.device, warmup_steps: int, d_model: int, max_grad_norm: float = 1.0):
        """
        Initializes the Trainer class.

        Args:
        ----
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer used to update the model parameters.
        loss_fn (Callable): The loss function used to calculate the loss.
        device (torch.device): The device used to train the model.
        warmup_steps (int): The number of warmup steps.
        d_model (int): The dimension of the model.
        max_grad_norm (float, optional): The maximum gradient norm. Defaults to 1.0.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.max_grad_norm = max_grad_norm
        self.step_num = 0
        self.scheduler = lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """
        Calculates the learning rate lambda.

        Args:
        ----
        step (int): The current step number.

        Returns:
        -------
        float: The learning rate lambda.
        """
        self.step_num = step
        return self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        """
        Trains the model for one epoch.

        Args:
        ----
        dataloader (DataLoader): The dataloader used to train the model.

        Returns:
        -------
        float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validates the model on a given dataloader.

        Args:
        ----
        dataloader (DataLoader): The dataloader used to validate the model.

        Returns:
        -------
        float: The average loss for the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.loss_fn(outputs, targets).item()
        return total_loss / len(dataloader)

    def should_stop_early(self, val_loss: float) -> bool:
        """
        Checks if the training should stop early.

        Args:
        ----
        val_loss (float): The validation loss.

        Returns:
        -------
        bool: True if the training should stop early, False otherwise.
        """
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = float('inf')
            self.epochs_no_improve = 0
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
            return False
        self.epochs_no_improve += 1
        return self.epochs_no_improve >= 5

    def save_checkpoint(self, path: str) -> None:
        """
        Saves the model checkpoint.

        Args:
        ----
        path (str): The path to save the checkpoint.
        """
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    # Define a dummy model, loss function, and optimizer
    model = nn.Linear(5, 5)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Create a dummy dataset and dataloader
    dataset = [(torch.randn(5), torch.randn(5)) for _ in range(100)]
    dataloader = DataLoader(dataset, batch_size=10)

    # Create a Trainer instance
    trainer = Trainer(model, optimizer, loss_fn, torch.device('cpu'), warmup_steps=100, d_model=5)

    # Train the model for 10 epochs
    for epoch in range(10):
        loss = trainer.train_one_epoch(dataloader)
        print(f'Epoch {epoch+1}, Loss: {loss}')

    # Validate the model
    val_loss = trainer.validate(dataloader)
    print(f'Validation Loss: {val_loss}')