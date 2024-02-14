import torch
from torch import Tensor, nn


class Arnold2018DeepModel(nn.Module):
    """The following class reimplements
    M. Arnold, *et al.,* "On Deep Learning-based Massive MIMO Indoor User Localization", 2019

    arXiv: https://arxiv.org/abs/1804.04826

    Not 100% sure if it's correctly implemented.
    """

    def __init__(self, in_features, n_outputs=2, random_state=None) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_outputs),
        )

        generator = torch.Generator().manual_seed(random_state)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu", generator=generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class Arnold2019SoundingModel(nn.Module):
    """The following class reimplements
    M. Arnold, *et al.,* "Novel Massive MIMO Channel Sounding Data Applied to Deep Learning-based Indoor Positioning", 2019

    arXiv: https://arxiv.org/abs/1810.04126

    Not 100% sure if it's correctly implemented.
    """

    def __init__(self, in_channels=2, n_outputs=2, random_state: int = None) -> None:
        super().__init__()

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

        generator = torch.Generator().manual_seed(random_state)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He initialization for Conv2d layers with ReLU activation
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu", generator=generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # Kaiming/He initialization for Linear layers
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu", generator=generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, h: Tensor, snr: Tensor) -> Tensor:
        return self.net(h)


# class Cerar2024Simple(nn.Module):
#     def __init__(self, in_features, n_outputs=2, random_state=None) -> None:
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(in_features, 8),
#             nn.LeakyReLU(),
#             nn.Dropout(),

#             nn.Linear(8, 8),
#             nn.LeakyReLU(),
#             nn.Dropout(),

#             nn.Linear(8, 8),
#             nn.LeakyReLU(),
#             nn.Dropout(),

#             nn.Linear(8, n_outputs),
#         )

#         generator = torch.Generator().manual_seed(random_state)
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu", generator=generator)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     #@torch.autocast(device_type='cuda', dtype=torch.bfloat16)
#     def forward(self, inputs: Tensor) -> Tensor:
#         h = inputs["h"]
#         return self.net(h)
