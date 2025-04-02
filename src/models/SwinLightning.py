"""
File: src/models/SwinUnetLightning.py

This Lightning module wraps the extended Swin-Unet architecture into the
generic training pipeline provided by BaseModel. All training, validation,
and test steps are inherited from BaseModel. Model-specific parameters (e.g.
img_size, patch_size, num_classes, embed_dim, depths, num_heads, window_size,
mlp_ratio, dropout, pretrained_path, learning_rate) are expected to be provided
via the YAML configuration files in the cfgs folder.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

# Import the generic BaseModel which implements training, validation, and test logic.
from .BaseModel import BaseModel
# Import the extended Swin-Unet architecture.
from .swin_models.newswinarchi import SwinUnet


class SwinUnetLightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: str,
        use_doy: bool = False,
        # required_img_size: Optional[Tuple[int, int]] = None,
        img_size: int =  224,
        patch_size: int =  4,
        num_classes: int =  1,
        embed_dim: int = 96,
        depths: Tuple[int, int, int, int] = [2, 2, 2, 2],
        num_heads: Tuple[int, int, int, int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pretrained_path: str = 'pretrained_ckpt\swin_tiny_patch4_window7_224.pth',
        *args: Any,
        **kwargs: Any,
    ):
        """
        Lightning module for extended Swin-Unet.
        
        Args:
            n_channels (int): Number of input channels.
            flatten_temporal_dimension (bool): Whether to flatten the temporal dimension.
            pos_class_weight (float): Positive class weight used in the loss.
            loss_function (str): Which loss function to use (e.g., "BCE", "Focal", etc.).
            use_doy (bool, optional): Whether to use day-of-year as an additional feature. Defaults to False.
            required_img_size (Optional[Tuple[int, int]], optional): Required image size for inference.
            *args, **kwargs: Additional arguments passed to BaseModel.
        
        Expected hparams (loaded via YAML) include:
            - img_size: int
            - patch_size: int
            - num_classes: int
            - embed_dim: int
            - depths: list
            - num_heads: list
            - window_size: int
            - mlp_ratio: float
            - dropout: float
            - pretrained_path: str (or empty string if not used)
            - learning_rate: float
        """
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=use_doy,
            # required_img_size=required_img_size,
            *args,
            **kwargs,
        )
        self.save_hyperparameters()

        print("hparams", self.hparams)
        # Instantiate the extended Swin-Unet using parameters from the YAML config (self.hparams).
        self.model = SwinUnet(
            img_size=self.hparams.img_size,
            patch_size=self.hparams.patch_size,
            in_chans=n_channels,
            num_classes=self.hparams.num_classes,
            embed_dim=self.hparams.embed_dim,
            depths=self.hparams.depths,
            num_heads=self.hparams.num_heads,
            window_size=self.hparams.window_size,
            mlp_ratio=self.hparams.mlp_ratio,
            dropout=self.hparams.dropout,
            pretrained_path=self.hparams.pretrained_path,
        )

    def forward(self, x, doys=None):
        """
        Forward pass through the Swin-Unet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, n_channels, H, W).
            doys: Not used for this model.
        
        Returns:
            torch.Tensor: Segmentation logits.
        """
        return self.model(x)
