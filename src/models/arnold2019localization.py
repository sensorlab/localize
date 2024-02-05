import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

from torch.nn import functional as F


import lightning as L

from sklearn.base import BaseEstimator, RegressorMixin


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
        # TODO: What if I introduce, to use small chunk of train data for validation?

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
            # max_epochs=self.max_epochs,
            max_epochs=1,
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

        return predictions.float().numpy()

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


def _initialize_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        # Kaiming/He initialization for Conv2d layers with ReLU activation
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # Kaiming/He initialization for Linear layers
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.constant_(m.bias, 0)


# class _CustomDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
#         """
#         Args:
#             X (numpy.ndarray): Features data as a NumPy array.
#             y (numpy.ndarray): Labels data as a NumPy array.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.X = torch.from_numpy(X).float()  # Convert features to PyTorch tensors
#         self.y = torch.from_numpy(y).float()  # Convert targets to PyTorch tensors
#         self.transform = transform

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         sample = self.X[idx], self.y[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

# (2, 16, 924)


class Arnold2019LocalizationModel(L.LightningModule):
    def __init__(self, in_channels=2, n_outputs=2) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # 2 x 16 x 924
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), padding=0, count_include_pad=True),
            # 32 x 16 x 231
            nn.ZeroPad2d(padding=(0, 1)),
            # 32 x 16 x 232
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), padding=0, count_include_pad=True),
            # 32 x 16 x 58
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(32 * 16 * 58, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        inputs, *_ = batch
        outputs = self(inputs)
        return outputs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# def prepare_model():
#     model = Arnold2019LocalizationModel()

#     return model


# def fit(model, features, targets, train_indices, test_indices):
#     # Initialize/reset model's weights
#     model.apply(_initialize_weights)

#     dataset = _CustomDataset(features, targets)
#     trainset = Subset(dataset, train_indices)
#     testset = Subset(dataset, test_indices)

#     n_jobs = joblib.cpu_count()

#     trainloader = DataLoader(
#         trainset,
#         batch_size=64,
#         shuffle=True,
#         num_workers=n_jobs,
#         pin_memory=True,
#     )

#     testloader = DataLoader(
#         testset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=n_jobs,
#         pin_memory=False,
#     )

#     #device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     assert torch.cuda.is_available()

#     trainer = L.Trainer(
#         accelerator='gpu',
#         precision='bf16-mixed',
#         max_epochs=100,
#         benchmark=True,
#         logger=False,
#         enable_checkpointing=False,
#     )

#     trainer.fit(model, trainloader, testloader)

#     return model


# def predict(model, features, targets, test_indices=None):

#     testset = _CustomDataset(features, targets)

#     if test_indices:
#         testset = Subset(testset, test_indices)

#     testloader = DataLoader(
#         testset,
#         batch_size=64,
#         shuffle=False,
#         num_workers=joblib.cpu_count(),
#         pin_memory=False,
#     )

#     trainer = L.Trainer(
#         accelerator='gpu',
#         precision='bf16-mixed',
#         max_epochs=100,
#         benchmark=True,
#         logger=False,
#         enable_checkpointing=False,
#     )

#     predictions = trainer.predict(model, testloader)
#     predictions = torch.cat(predictions, dim=0)
#     predictions = predictions.numpy()

#     return predictions


if __name__ == "__main__":
    from torchinfo import summary

    net = Arnold2019LocalizationModel()
    print(summary(net, (100, 2, 16, 924)))
