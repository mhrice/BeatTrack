import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class BeatNet(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.ConvBlock = ConvBlock()
        self.TCN = TCN(n_blocks=11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ConvBlock(x)  # b, 16, conv, 1
        # Remove last dimension
        x = x.view(-1, x.shape[1], x.shape[2])
        x = self.TCN(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="test")
        return loss

    def common_step(self, batch, batch_idx, mode: str = "train"):
        specs, labels = batch
        import pdb

        pdb.set_trace()
        preds = self(specs)
        loss = F.binary_cross_entropy(preds.squeeze(1), labels.float())

        self.log(f"{mode}_loss", loss)

        return loss


class ConvBlock(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 0),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 0),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(1, 8),
        )
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.elu(x)
        x = self.dropout(x)
        return x


class TCN(nn.Module):
    def __init__(
        self,
        n_blocks: int,
    ):
        super().__init__()
        # self.input_conv = nn.Conv1d()
        self.resblocks = nn.ModuleList(
            [
                Resblock(
                    in_channels=16,
                    kernel_size=5,
                    dilation=2**i,
                    dropout=0.1,
                    num_filters=16,
                )
                for i in range(n_blocks)
            ]
        )
        self.activation1 = nn.ELU()
        self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1)
        self.activation2 = nn.ELU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        # x = self.input_conv(x)
        for block in self.resblocks:
            x = block(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        return x


class Resblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        num_filters: int,
    ):
        super().__init__()
        self.dialated_conv1 = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2 * dilation,
                dilation=dilation,
            )
        )
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dialated_conv2 = weight_norm(
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=kernel_size // 2 * dilation,
            )
        )
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.skip = nn.Conv1d(
            in_channels=in_channels, out_channels=num_filters, kernel_size=1
        )
        self._initialise_weights(self.dialated_conv1, self.dialated_conv2, self.skip)

    def forward(self, x):
        y = self.dialated_conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.dialated_conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        skip = self.skip(x)
        return y + skip

    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)
