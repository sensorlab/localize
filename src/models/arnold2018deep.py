"""The following script reimplements
M. Arnold, *et al.,* "On Deep Learning-based Massive MIMO Indoor User Localization", 2019

arXiv: https://arxiv.org/abs/1804.04826

Not 100% sure if it's correctly implemented.
"""
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

import lightning as L


class Arnold2018DeepModel(L.LightningModule):
    def __init__(self, in_channels=16 * 924 * 2, n_outputs=2, random_state: int = None) -> None:
        super().__init__()
        self.random_state = random_state

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_outputs),
        )

        self.reset_weights()

    def reset_weights(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Kaiming/He initialization for Conv2d layers with ReLU activation
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


if __name__ == "__main__":
    pass

    # net = Arnold2018DeepModel()
    # print(summary(net, (100, 2, 16, 924)))
