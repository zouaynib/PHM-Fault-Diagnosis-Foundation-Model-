"""
Baseline CNN Model — Per-Dataset 1D CNN for Fault Classification
================================================================
Architecture:
  - Stack of Conv1D -> BatchNorm -> ReLU -> Dropout blocks
  - AdaptiveAvgPool1D
  - Fully connected classifier
  - Optional RUL regression head for fair comparison with foundation model
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.relu(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    """
    1D CNN for single-dataset fault classification with optional RUL head.

    Supports multivariate input of shape (B, C, L) where C is the number
    of input channels (sensor streams).
    """
    def __init__(self, num_classes, window_length=2560, in_channels=1,
                 channels=(32, 64, 128, 128, 64),
                 kernel_size=7, dropout=0.3,
                 has_rul_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.has_rul_head = has_rul_head

        # ---------- feature backbone ----------
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch, kernel_size, dropout))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ---------- classification head ----------
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(channels[-1], 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = None

        # ---------- optional RUL regression head ----------
        if has_rul_head:
            self.rul_head = nn.Sequential(
                nn.Linear(channels[-1], 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.rul_head = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (B, C, L)
            Multivariate time-series input.

        Returns
        -------
        cls_logits : Tensor of shape (B, num_classes) or None
            Classification logits.  None when num_classes == 0.
        rul_pred : Tensor of shape (B,) or None
            Normalised RUL prediction.  None when has_rul_head is False.
        """
        h = self.features(x)              # (B, channels[-1], L')
        h = self.pool(h).squeeze(-1)      # (B, channels[-1])

        cls_logits = self.classifier(h) if self.classifier is not None else None
        rul_pred = self.rul_head(h).squeeze(-1) if self.rul_head is not None else None

        return cls_logits, rul_pred


if __name__ == "__main__":
    # --- classification-only, single-channel ---
    model = BaselineCNN(num_classes=4, window_length=2560, in_channels=1)
    x = torch.randn(8, 1, 2560)
    cls, rul = model(x)
    print(f"[cls-only, 1ch]  Input: {x.shape} -> cls: {cls.shape}, rul: {rul}")

    # --- classification + RUL, multivariate (3 channels) ---
    model_mv = BaselineCNN(num_classes=4, window_length=2560, in_channels=3,
                           has_rul_head=True)
    x_mv = torch.randn(8, 3, 2560)
    cls_mv, rul_mv = model_mv(x_mv)
    print(f"[cls+rul, 3ch]   Input: {x_mv.shape} -> cls: {cls_mv.shape}, rul: {rul_mv.shape}")

    # --- RUL-only, multivariate (6 channels) ---
    model_rul = BaselineCNN(num_classes=0, window_length=2560, in_channels=6,
                            has_rul_head=True)
    x_rul = torch.randn(8, 6, 2560)
    cls_r, rul_r = model_rul(x_rul)
    print(f"[rul-only, 6ch]  Input: {x_rul.shape} -> cls: {cls_r}, rul: {rul_r.shape}")

    print(f"\nParameters (cls-only):  {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parameters (cls+rul):   {sum(p.numel() for p in model_mv.parameters()):,}")
    print(f"Parameters (rul-only):  {sum(p.numel() for p in model_rul.parameters()):,}")
