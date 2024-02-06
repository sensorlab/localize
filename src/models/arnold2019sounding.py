"""The following script reimplements
M. Arnold, *et al.,* "Novel Massive MIMO Channel Sounding Data Applied to Deep Learning-based Indoor Positioning", 2019

arXiv: https://arxiv.org/abs/1810.04126

Not 100% sure if it's correctly implemented.
"""
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

import lightning as L


class Arnold2019SoundingModel(L.LightningModule):
    def __init__(self, in_channels=2, n_outputs=2, random_state: int = None) -> None:
        super().__init__()
        self.random_state = random_state

        self.net = nn.Sequential(
            # 2 x 16 x 924
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), padding=0, count_include_pad=True),
            # Fix the padding
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

        self.reset_weights()

    def reset_weights(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He initialization for Conv2d layers with ReLU activation
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Kaiming/He initialization for Linear layers
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
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
    from torchinfo import summary

    net = Arnold2019SoundingModel()
    print(summary(net, (100, 2, 16, 924)))
