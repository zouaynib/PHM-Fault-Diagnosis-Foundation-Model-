"""
Foundation Model — PatchTST-Style Channel-Independent Transformer Encoder
=========================================================================
Architecture:
  - Channel-independent processing: each channel goes through the SAME
    transformer backbone, then outputs are averaged across actual channels.
  - PatchEmbedding: unfold 1-D signal into patches, project via Linear.
  - Learnable positional encoding added to patch tokens.
  - Pre-norm TransformerEncoder (norm_first=True, batch_first=True).
  - Frequency embedding: log10(freq) -> MLP -> concatenated to backbone output.
  - Dataset ID embedding: learned embedding -> concatenated to backbone output.
  - Projector MLP maps to latent_dim.
  - Per-dataset classification heads and RUL regression heads.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Unfold a 1-D signal into non-overlapping or overlapping patches and
    project each patch to *d_model* dimensions."""

    def __init__(self, patch_size: int, patch_stride: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.value_embedding = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) — single-channel waveform.
        Returns:
            (B, num_patches, d_model)
        """
        # x.unfold(dimension, size, step) -> (B, num_patches, patch_size)
        x = x.unfold(1, self.patch_size, self.patch_stride)
        return self.value_embedding(x)


# ---------------------------------------------------------------------------
# Learnable Positional Encoding
# ---------------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embedding added to patch tokens."""

    def __init__(self, max_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_patches, d_model)"""
        return x + self.pos_embed[:, :x.shape[1], :]


# ---------------------------------------------------------------------------
# Frequency Embedding (kept from original)
# ---------------------------------------------------------------------------
class FrequencyEmbedding(nn.Module):
    """Embed sampling frequency via log10 -> MLP.
    Allows the model to be frequency-aware."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        """freq: (B,) scalar sampling frequencies -> (B, embed_dim)"""
        log_freq = torch.log10(freq.clamp(min=1e-3)).unsqueeze(-1)  # (B, 1)
        return self.mlp(log_freq)


# ---------------------------------------------------------------------------
# Foundation Model
# ---------------------------------------------------------------------------
class FoundationModel(nn.Module):
    """
    PatchTST-style channel-independent Transformer foundation model for
    multi-domain PHM time-series.
    """

    def __init__(
        self,
        dataset_configs,
        window_length: int = 2560,
        d_model: int = 128,
        patch_size: int = 64,
        patch_stride: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        freq_embed_dim: int = 32,
        dataset_embed_dim: int = 32,
        latent_dim: int = 128,
        max_channels: int = 14,
        use_freq_embed: bool = True,
        use_dataset_embed: bool = True,
    ):
        """
        Args:
            dataset_configs: list of dicts, each with keys:
                - 'name': str
                - 'num_channels': int
                - 'tasks': list of dicts with 'type' and optionally 'num_classes'
            window_length: signal length L (after windowing / padding).
            d_model: transformer hidden dimension.
            patch_size: number of time-steps per patch.
            patch_stride: stride between consecutive patches.
            num_heads: number of attention heads.
            num_layers: number of transformer encoder layers.
            dim_feedforward: FFN intermediate dimension.
            dropout: dropout rate used throughout.
            activation: activation function for transformer FFN.
            freq_embed_dim: dimension of the frequency embedding.
            dataset_embed_dim: dimension of the dataset embedding.
            latent_dim: output latent dimension (after projector).
            max_channels: maximum number of channels across all datasets.
            use_freq_embed: whether to use frequency embedding.
            use_dataset_embed: whether to use dataset embedding.
        """
        super().__init__()

        self.use_freq_embed = use_freq_embed
        self.use_dataset_embed = use_dataset_embed
        self.num_datasets = len(dataset_configs)
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_channels = max_channels
        self.dataset_configs = dataset_configs

        # --- Patch embedding --------------------------------------------------
        self.patch_embed = PatchEmbedding(patch_size, patch_stride, d_model)

        # Compute max number of patches for positional encoding
        num_patches = (window_length - patch_size) // patch_stride + 1
        self.pos_encode = LearnablePositionalEncoding(num_patches, d_model)

        # --- Transformer encoder (pre-norm) -----------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

        # --- Embeddings -------------------------------------------------------
        feat_dim = d_model
        if use_freq_embed:
            self.freq_embed = FrequencyEmbedding(freq_embed_dim)
            feat_dim += freq_embed_dim
        if use_dataset_embed:
            self.ds_embed = nn.Embedding(self.num_datasets, dataset_embed_dim)
            feat_dim += dataset_embed_dim

        # --- Projector MLP ----------------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Per-dataset task heads -------------------------------------------
        self.cls_heads = nn.ModuleDict()
        self.rul_heads = nn.ModuleDict()

        for ds_id, ds_cfg in enumerate(dataset_configs):
            for task in ds_cfg.get("tasks", []):
                if task["type"] == "classification":
                    num_classes = task["num_classes"]
                    self.cls_heads[f"cls_{ds_id}"] = nn.Linear(latent_dim, num_classes)
                elif task["type"] == "regression":
                    self.rul_heads[f"rul_{ds_id}"] = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.GELU(),
                        nn.Linear(latent_dim, 1),
                        nn.Sigmoid(),
                    )

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    def forward_backbone(
        self, x: torch.Tensor, num_channels: torch.Tensor
    ) -> torch.Tensor:
        """
        Channel-independent transformer backbone.

        Args:
            x: (B, C_max, L) — zero-padded multivariate signal.
            num_channels: (B,) — actual number of channels per sample.

        Returns:
            (B, d_model) — aggregated representation.
        """
        B, C_max, L = x.shape
        device = x.device

        unique_nch = torch.unique(num_channels)

        # Common fast path: all samples have the same number of channels
        if unique_nch.numel() == 1:
            C = unique_nch.item()
            # (B, C, L) -> (B*C, L)
            xc = x[:, :C, :].reshape(B * C, L)
            # Patch embed -> (B*C, num_patches, d_model)
            tokens = self.patch_embed(xc)
            tokens = self.pos_encode(tokens)
            # Transformer -> (B*C, num_patches, d_model)
            tokens = self.transformer(tokens)
            tokens = self.norm(tokens)
            # Mean pool over patches -> (B*C, d_model)
            pooled = tokens.mean(dim=1)
            # Reshape -> (B, C, d_model) -> mean over channels -> (B, d_model)
            pooled = pooled.view(B, C, self.d_model).mean(dim=1)
            return pooled

        # Slow path: mixed channel counts — group by unique num_channels
        out = torch.zeros(B, self.d_model, device=device)

        for nch in unique_nch:
            nch_int = nch.item()
            mask = num_channels == nch  # (B,)
            idx = mask.nonzero(as_tuple=True)[0]
            B_sub = idx.shape[0]

            # (B_sub, C_max, L) -> (B_sub, nch_int, L)
            xc = x[idx, :nch_int, :]
            # -> (B_sub * nch_int, L)
            xc = xc.reshape(B_sub * nch_int, L)

            tokens = self.patch_embed(xc)
            tokens = self.pos_encode(tokens)
            tokens = self.transformer(tokens)
            tokens = self.norm(tokens)
            pooled = tokens.mean(dim=1)  # (B_sub * nch_int, d_model)
            pooled = pooled.view(B_sub, nch_int, self.d_model).mean(dim=1)

            out[idx] = pooled

        return out

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        freq: torch.Tensor,
        dataset_id: torch.Tensor,
        num_channels: torch.Tensor,
    ):
        """
        Args:
            x: (B, C_max, L)
            freq: (B,) sampling frequencies
            dataset_id: (B,) integer dataset IDs
            num_channels: (B,) actual channel count per sample

        Returns:
            cls_outputs: dict {ds_id_int: (N_ds, num_classes)}
            rul_outputs: dict {ds_id_int: (N_ds,)}
            latent:      (B, latent_dim)
        """
        backbone_feat = self.forward_backbone(x, num_channels)  # (B, d_model)

        parts = [backbone_feat]
        if self.use_freq_embed:
            parts.append(self.freq_embed(freq))
        if self.use_dataset_embed:
            parts.append(self.ds_embed(dataset_id))

        feat = torch.cat(parts, dim=-1)  # (B, feat_dim)
        latent = self.projector(feat)      # (B, latent_dim)

        # Route each sample to its dataset-specific head(s)
        unique_ds = torch.unique(dataset_id)
        cls_outputs = {}
        rul_outputs = {}

        for ds_id in unique_ds:
            ds_id_int = ds_id.item()
            mask = dataset_id == ds_id
            ds_latent = latent[mask]

            cls_key = f"cls_{ds_id_int}"
            if cls_key in self.cls_heads:
                cls_outputs[ds_id_int] = self.cls_heads[cls_key](ds_latent)

            rul_key = f"rul_{ds_id_int}"
            if rul_key in self.rul_heads:
                rul_outputs[ds_id_int] = self.rul_heads[rul_key](ds_latent).squeeze(-1)

        return cls_outputs, rul_outputs, latent

    # ------------------------------------------------------------------
    # Single-dataset convenience
    # ------------------------------------------------------------------
    def forward_single_dataset(
        self,
        x: torch.Tensor,
        freq: torch.Tensor,
        dataset_id_int: int,
        num_channels_int: int,
    ):
        """Convenience method when all samples come from the same dataset.

        Returns:
            (cls_logits_or_None, rul_preds_or_None)
        """
        B = x.shape[0]
        device = x.device
        freq_t = freq if isinstance(freq, torch.Tensor) else torch.full((B,), freq, device=device)
        dsid_t = torch.full((B,), dataset_id_int, dtype=torch.long, device=device)
        nch_t = torch.full((B,), num_channels_int, dtype=torch.long, device=device)

        cls_outputs, rul_outputs, latent = self(x, freq_t, dsid_t, nch_t)

        cls_logits = cls_outputs.get(dataset_id_int, None)
        rul_preds = rul_outputs.get(dataset_id_int, None)
        return cls_logits, rul_preds

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------
    def get_backbone_params(self):
        """Transformer encoder + patch embedding + positional encoding params."""
        params = []
        params.extend(self.patch_embed.parameters())
        params.extend(self.pos_encode.parameters())
        params.extend(self.transformer.parameters())
        params.extend(self.norm.parameters())
        return params

    def get_head_params(self, ds_id=None):
        """Classification + RUL head params, optionally filtered by dataset."""
        params = []
        if ds_id is not None:
            cls_key = f"cls_{ds_id}"
            rul_key = f"rul_{ds_id}"
            if cls_key in self.cls_heads:
                params.extend(self.cls_heads[cls_key].parameters())
            if rul_key in self.rul_heads:
                params.extend(self.rul_heads[rul_key].parameters())
        else:
            for head in self.cls_heads.values():
                params.extend(head.parameters())
            for head in self.rul_heads.values():
                params.extend(head.parameters())
        return params

    def get_embed_params(self):
        """Frequency + dataset embedding params."""
        params = []
        if self.use_freq_embed:
            params.extend(self.freq_embed.parameters())
        if self.use_dataset_embed:
            params.extend(self.ds_embed.parameters())
        return params


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset configs matching the 5 datasets
    ds_configs = [
        {  # 0: CWRU
            "name": "CWRU",
            "num_channels": 1,
            "tasks": [{"type": "classification", "num_classes": 4}],
        },
        {  # 1: PRONOSTIA
            "name": "PRONOSTIA",
            "num_channels": 2,
            "tasks": [{"type": "regression"}],
        },
        {  # 2: CMAPSS
            "name": "CMAPSS",
            "num_channels": 14,
            "tasks": [
                {"type": "classification", "num_classes": 2},
                {"type": "regression"},
            ],
        },
        {  # 3: Paderborn
            "name": "Paderborn",
            "num_channels": 1,
            "tasks": [{"type": "classification", "num_classes": 3}],
        },
        {  # 4: XJTU-SY
            "name": "XJTU-SY",
            "num_channels": 2,
            "tasks": [
                {"type": "classification", "num_classes": 3},
                {"type": "regression"},
            ],
        },
    ]

    window_length = 2560
    max_channels = 14

    model = FoundationModel(
        ds_configs,
        window_length=window_length,
        d_model=128,
        patch_size=64,
        patch_stride=32,
        num_heads=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        freq_embed_dim=32,
        dataset_embed_dim=32,
        latent_dim=128,
        max_channels=max_channels,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"Cls heads: {list(model.cls_heads.keys())}")
    print(f"RUL heads: {list(model.rul_heads.keys())}")

    # --- Test each dataset individually ---
    print("\n--- Single-dataset forward tests ---")
    dataset_specs = [
        # (ds_id, name, num_ch, batch_size, freq)
        (0, "CWRU", 1, 8, 12000.0),
        (1, "PRONOSTIA", 2, 6, 25600.0),
        (2, "CMAPSS", 14, 4, 1.0),
        (3, "Paderborn", 1, 8, 64000.0),
        (4, "XJTU-SY", 2, 6, 25600.0),
    ]

    for ds_id, name, n_ch, bs, fs in dataset_specs:
        x = torch.randn(bs, max_channels, window_length)
        # Zero-out unused channels
        if n_ch < max_channels:
            x[:, n_ch:, :] = 0.0
        freq = torch.full((bs,), fs)
        cls_logits, rul_preds = model.forward_single_dataset(x, freq, ds_id, n_ch)
        cls_str = f"cls={cls_logits.shape}" if cls_logits is not None else "cls=None"
        rul_str = f"rul={rul_preds.shape}" if rul_preds is not None else "rul=None"
        print(f"  {name:12s} (ds_id={ds_id}, ch={n_ch:2d}, B={bs}): {cls_str}, {rul_str}")

    # --- Test mixed-batch forward ---
    print("\n--- Mixed-batch forward test ---")
    # Build a batch with samples from all 5 datasets
    batch_sizes = [4, 3, 2, 4, 3]  # per dataset
    total_B = sum(batch_sizes)
    x_all = torch.randn(total_B, max_channels, window_length)
    freq_all = torch.zeros(total_B)
    dsid_all = torch.zeros(total_B, dtype=torch.long)
    nch_all = torch.zeros(total_B, dtype=torch.long)

    offset = 0
    for ds_id, name, n_ch, bs, fs in dataset_specs:
        bs_actual = batch_sizes[ds_id]
        x_all[offset : offset + bs_actual, n_ch:, :] = 0.0
        freq_all[offset : offset + bs_actual] = fs
        dsid_all[offset : offset + bs_actual] = ds_id
        nch_all[offset : offset + bs_actual] = n_ch
        offset += bs_actual

    cls_outputs, rul_outputs, latent = model(x_all, freq_all, dsid_all, nch_all)
    print(f"  Latent: {latent.shape}")
    for k, v in cls_outputs.items():
        print(f"  cls[ds={k}]: {v.shape}")
    for k, v in rul_outputs.items():
        print(f"  rul[ds={k}]: {v.shape}")

    # --- Verify parameter groups ---
    print("\n--- Parameter group counts ---")
    n_backbone = sum(p.numel() for p in model.get_backbone_params())
    n_heads = sum(p.numel() for p in model.get_head_params())
    n_embeds = sum(p.numel() for p in model.get_embed_params())
    n_proj = sum(p.numel() for p in model.projector.parameters())
    print(f"  Backbone (transformer+patch+pos+norm): {n_backbone:,}")
    print(f"  Heads (cls+rul):                       {n_heads:,}")
    print(f"  Embeddings (freq+dataset):             {n_embeds:,}")
    print(f"  Projector:                             {n_proj:,}")
    print(f"  Sum:   {n_backbone + n_heads + n_embeds + n_proj:,}")
    print(f"  Total: {n_params:,}")

    print("\nAll tests passed.")
