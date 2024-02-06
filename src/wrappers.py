from sklearn.base import BaseEstimator, RegressorMixin
import lightning as L
import joblib

from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.nn import functional as F


class LightningRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model: L.LightningModule,
        batch_size: int = 64,
        max_epochs: int = 100,
        memory: joblib.Memory = None,
        random_state: int = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.memory = memory
        self.random_state = random_state

        self.n_jobs = joblib.cpu_count(only_physical_cores=True)

        self._fit_cached = memory.cache(self._fit) if memory else self._fit

    def _fit(self, X, y):
        # TODO: What if I introduce use of small chunk of train data for validation?
        # Or to know when to stop, track progress?

        if hasattr(self.model, "reset_weights"):
            self.model.reset_weights()

        assert torch.cuda.is_available()
        self.model.train()

        torch.set_float32_matmul_precision("medium")

        # Convert data to PyTorch dataset
        dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            pin_memory=True,
        )

        # Train the model
        trainer = L.Trainer(
            accelerator="gpu",
            precision="bf16-mixed",
            max_epochs=self.max_epochs,
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(self.model, train_loader)

        return self

    def fit(self, X, y):
        return self._fit_cached(X, y)

    def predict(self, X):
        dataset = TensorDataset(torch.tensor(X).float())
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            pin_memory=False,
        )
        # Train the model
        trainer = L.Trainer(
            accelerator="gpu",
            precision="bf16-mixed",
            max_epochs=self.max_epochs,
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
        )

        predictions = trainer.predict(self.model, dataloader)
        predictions = torch.concat(predictions, dim=0)
        predictions = predictions.float().numpy()

        return predictions

        # Implement model inference using the trained model
        # self.model.eval()  # Set model to evaluation mode
        # X_tensor = torch.tensor(X).float()
        # with torch.inference_mode():
        #    predictions = self.model(X_tensor)
        # return predictions.numpy()

    def score(self, X, y):
        # Implement a scoring method, e.g., accuracy
        predictions = self.predict(X)
        loss = F.mse_loss(predictions, y)
        return loss


# def _initialize_weights(m: nn.Module) -> None:
#     if isinstance(m, nn.Conv2d):
#         # Kaiming/He initialization for Conv2d layers with ReLU activation
#         nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

#     elif isinstance(m, nn.Linear):
#         # Kaiming/He initialization for Linear layers
#         nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
#         nn.init.constant_(m.bias, 0)
