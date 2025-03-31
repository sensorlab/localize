# check if running on MacOS
import platform

import torch


if platform.system() == "Darwin":
    # check if M1/M2
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


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
            nn.Linear(32 * 16 * 58, 256),
            nn.LeakyReLU(),
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


# # def train_sklearn_model(model, features, targets, train_idx, test_idx):
# #    model.fit(features)

# import torch
# from torch import nn
# from torch.nn import functional as F
# from torchvision import models


# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )


# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class PirnatEco(nn.Module):
#     def __init__(self, norm_layer=None) -> None:
#         super().__init__()

#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d

#         self._norm_layer = norm_layer
#         self.dilation = 1
#         self.inplanes = 32
#         self.groups = 1
#         self.base_width = 64

#         self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=(1, 7), stride=(1, 3), padding=(1, 4))
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.maxpool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

#         self.layer1 = self._make_layer(models.resnet.BasicBlock, planes=32, blocks=2, stride=2, dilate=False)
#         self.layer2 = self._make_layer(models.resnet.BasicBlock, planes=64, blocks=2, stride=2, dilate=False)
#         self.layer3 = self._make_layer(models.resnet.BasicBlock, planes=128, blocks=2, stride=2, dilate=False)
#         self.layer4 = self._make_layer(models.resnet.BasicBlock, planes=256, blocks=2, stride=2, dilate=False)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flat = nn.Flatten()
#         self.fc1 = nn.Linear(256, 1000)
#         self.fc = nn.Linear(1000, 2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = self.flat(x)
#         x = self.fc1(x)
#         x = F.leaky_relu(x, negative_slope=1e-3)
#         x = self.fc(x)

#         return x

#     def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )

#         return nn.Sequential(*layers)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x


# def resnet18():
#     model = ResNet(BasicBlock, [2, 2, 2, 2])
#     return model


# # Example of using the model
# model = resnet18()
# print(model)


# if __name__ == "__main__":
#     model = resnet18()

#     from torchinfo import summary

#     summary(model, input_size=(500, 3, 256, 256))
