from sklearn.base import BaseEstimator, RegressorMixin
import lightning as L
import joblib

from typing import Any

from copy import copy, deepcopy

from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.nn import functional as F
from sklearn.exceptions import NotFittedError
import numpy as np
from skorch.dataset import Dataset

from collections import defaultdict


def filter_keys(kwargs: dict[str, Any], prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in kwargs.items() if k.startswith(prefix)}


class LightningSklearnRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, module, memory=None, **kwargs):
        self.model_class = module

        self.model = None

        self._init_kwargs = deepcopy(kwargs)
        self._kwargs = deepcopy(self._init_kwargs)

        self._dataloader_defaults = dict(
            batch_size=128,
            pin_memory=True,
            num_workers=4,
        )

        self._trainer_defaults = dict(
            accelerator="gpu",
            precision="bf16-mixed",
            max_epochs=100,
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
        )

        # Use joblib's cache, if user provides it.
        self._fit_cached = memory.cache(self._fit) if memory else self._fit

    def set_params(self, **params) -> BaseEstimator:
        self._kwargs = {**self._init_kwargs, **params}
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return deepcopy(self._kwargs) if deep else copy(self._kwargs)

    def _fit(self, X, y):
        # Convert X, y to PyTorch datasets
        dataset = Dataset(X, y)

        # TODO: What if I introduce use of small chunk of train data for validation?
        # Or to know when to stop, track progress?

        # Sanity checks
        assert torch.cuda.is_available()
        assert torch.cuda.is_bf16_supported()
        torch.set_float32_matmul_precision("medium")

        dataloader_kwargs = {**self._dataloader_defaults, **filter_keys(self.get_params(), "dataloader__")}
        trainloader = DataLoader(dataset=dataset, shuffle=True, **dataloader_kwargs)

        trainer_kwargs = {**self._trainer_defaults, **filter_keys(self.get_params(), "trainer__")}
        self.trainer = L.Trainer(**trainer_kwargs)

        model_kwargs = filter_keys(self.get_params(), "module__")
        self.model = self.model_class(**model_kwargs)

        self.trainer.fit(self.model, trainloader)

        return self

    def fit(self, X, y):
        return self._fit_cached(X, y)

    def predict(self, X):
        if not hasattr(self, "model"):
            raise NotFittedError(
                f"This {self.model_class.__name__} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        dataset = Dataset(X)

        dataloader_kwargs = {**self._dataloader_defaults, **filter_keys(self.get_params(), "dataloader__")}
        testloader = DataLoader(dataset, shuffle=False, **dataloader_kwargs)

        # Train the model
        predictions = self.trainer.predict(self.model, testloader)
        predictions = torch.concat(predictions, dim=0)
        predictions = predictions.float().numpy()

        return predictions

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))


class PyTorchLightningRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, batch_size, max_epochs, random_state=None, memory=None, **model_kwargs):
        self._model_class = model_class
        self.model_kwargs = model_kwargs

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.model_kwargs["random_state"] = random_state

        self._model = None

        # Use joblib's cache, if user provides it.
        self._fit_cached = memory.cache(self._fit) if memory else self._fit

        self._tags = {
            "multioutput": True,
            "stateless": False,
            "requires_fit": True,
            "preserves_dtype": [np.float64],
            "requires_y": True,  # TODO: Is this OK?
        }

    def get_params(self, deep=True):
        params = {"model_class": self.model_class, **self.kwargs}
        if deep:
            # In case of deep getting, include parameters from the model itself
            for key, value in self.model.named_parameters():
                params[f"model__{key}"] = value
        return params

    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def _fit(self, X, y):
        # TODO: What if I introduce use of small chunk of train data for validation?
        # Or to know when to stop, track progress?

        # Sanity checks
        assert torch.cuda.is_available()
        assert torch.cuda.is_bf16_supported()
        torch.set_float32_matmul_precision("medium")

        # Convert X, y to PyTorch datasets
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=joblib.cpu_count(),
            pin_memory=True,
        )

        self.model = self.model_class(**self.model_kwargs)

        # Train the model
        self.trainer = L.Trainer(
            accelerator="gpu",
            precision="bf16-mixed",
            max_epochs=self.max_epochs,
            benchmark=True,
            logger=False,
            enable_checkpointing=False,
        )

        self.trainer.fit(self.model, trainloader)

        return self

    def fit(self, X, y):
        return self._fit_cached(X, y)

    def predict(self, X):
        if not self.model:
            raise NotFittedError(
                f"This {self.model_class.__name__} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        testloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=joblib.cpu_count(),
            pin_memory=False,
        )

        # Train the model
        predictions = self.trainer.predict(self.model, testloader)
        predictions = torch.concat(predictions, dim=0)
        predictions = predictions.float().numpy()

        return predictions

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X))

    def _get_tags(self):
        return self._tags


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
