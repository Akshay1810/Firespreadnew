import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# Utility functions
##############################################

def sequence_to_image(x, H, W):
    """
    Reshape a sequence (B, L, C) into an image tensor (B, C, H, W).
    """
    B, L, C = x.shape
    assert L == H * W, "Sequence length does not match H*W"
    x = x.view(B, H, W, C).permute(0, 3, 1, 2)
    return x

def window_partition(x, window_size):
    """
    Partition input feature map x into non-overlapping windows.
    Input x: (B, H, W, C)
    Returns: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape

    # Compute padding amounts for H and W
    pad_H = (window_size - (H % window_size)) % window_size
    pad_W = (window_size - (W % window_size)) % window_size
    if pad_H != 0 or pad_W != 0:
        # F.pad for channel-last expects padding in the order: (pad_C_left, pad_C_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
        x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H))
        H, W = H + pad_H, W + pad_W

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partition process.
    Input windows: (num_windows*B, window_size*window_size, C)
    H, W: original (unpadded) spatial height and width.
    Returns: (B, H, W, C)
    """
    # Compute the padding amounts as in window_partition.
    pad_H = (window_size - (H % window_size)) % window_size
    pad_W = (window_size - (W % window_size)) % window_size
    H_pad, W_pad = H + pad_H, W + pad_W

    # Calculate batch size using padded dimensions.
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    
    # Reshape using padded dimensions.
    x = windows.reshape(B, H_pad // window_size, W_pad // window_size,
                          window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
    
    # Crop to the original H and W.
    if pad_H or pad_W:
        x = x[:, :H, :W, :]
    return x

##############################################
# Patch Embedding & Merging
##############################################

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        """
        Splits the image into patches and projects them into an embedding space.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, in_chans, H, W)
        print('A', x.shape, self.proj(x).shape)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """
        Downsamples the feature map by merging patches.
        Args:
            input_resolution (tuple): (H, W) resolution of the input feature map.
            dim (int): Number of input channels.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        Returns:
            x: (B, (H/2)*(W/2), 2*C)
            New H, W values.
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        x = x.view(B, H, W, C)
        # If H or W are odd, pad so that they are even.
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

##############################################
# Swin Transformer Block
##############################################

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.):
        """
        A simplified Swin Transformer block.
        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Size of cyclic shift. (0 means no shift)
            mlp_ratio (float): Ratio for the hidden dimension in the MLP.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, H, W):
        """
        x: (B, L, C) with L = H * W.
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Apply cyclic shift if shift_size > 0.
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows.
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, window_size*window_size, C)
        # Prepare for multi-head attention: (L_window, B_windows, C)
        x_windows = x_windows.transpose(0, 1)
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
        attn_windows = attn_windows.transpose(0, 1)
        # Merge windows.
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift.
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.reshape(B, L, C)
        # Add skip connection and apply MLP.
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

##############################################
# BasicLayer: Encoder Stage
##############################################

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio, dropout, downsample=None):
        """
        A stage of Swin Transformer blocks.
        Args:
            dim (int): Input dimension.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Attention window size.
            mlp_ratio (float): MLP ratio.
            dropout (float): Dropout probability.
            downsample (nn.Module or None): Downsampling module (e.g. PatchMerging).
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for i in range(depth)
        ])
        self.downsample = downsample
    
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        return x, H, W

##############################################
# Decoder Block for Skip Connection Fusion
##############################################

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        A decoder block that fuses upsampled features with skip connections.
        Args:
            in_channels (int): Number of channels of the upsampled feature.
            skip_channels (int): Number of channels of the skip connection.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

##############################################
# Extended Swin-Unet Model
##############################################

class SwinUnet(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1,
                 embed_dim=96,
                 depths=[2, 2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 dropout=0.,
                 pretrained_path=None):
        """
        Extended Swin-Unet with hierarchical encoder (with patch merging) and a symmetric decoder.
        
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of segmentation classes.
            embed_dim (int): Embedding dimension.
            depths (list): Number of transformer blocks in each stage.
            num_heads (list): Number of attention heads per stage.
            window_size (int): Window size for attention.
            mlp_ratio (float): MLP expansion ratio.
            dropout (float): Dropout probability.
            pretrained_path (str, optional): Path to pretrained weights.
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # Calculate resolution after patch embedding.
        H0 = img_size // patch_size
        W0 = img_size // patch_size

        # Encoder: 4 stages with increasing channel dimensions.
        self.layer1 = BasicLayer(dim=embed_dim,
                                  depth=depths[0],
                                  num_heads=num_heads[0],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  dropout=dropout,
                                  downsample=None)
        self.layer2 = BasicLayer(dim=embed_dim,
                                  depth=depths[1],
                                  num_heads=num_heads[1],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  dropout=dropout,
                                  downsample=PatchMerging((H0, W0), embed_dim))
        self.layer3 = BasicLayer(dim=embed_dim * 2,
                                  depth=depths[2],
                                  num_heads=num_heads[2],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  dropout=dropout,
                                  downsample=PatchMerging((H0 // 2, W0 // 2), embed_dim * 2))
        self.layer4 = BasicLayer(dim=embed_dim * 4,
                                  depth=depths[3],
                                  num_heads=num_heads[3],
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  dropout=dropout,
                                  downsample=PatchMerging((H0 // 4, W0 // 4), embed_dim * 4))
        
        # Decoder: Upsample and fuse with skip connections.
        # Note: Channel dimensions from encoder:
        # layer1: embed_dim, layer2: embed_dim*2, layer3: embed_dim*4, layer4: embed_dim*8.
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DecoderBlock(in_channels=embed_dim * 8, skip_channels=embed_dim * 4, out_channels=embed_dim * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DecoderBlock(in_channels=embed_dim * 4, skip_channels=embed_dim * 2, out_channels=embed_dim * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DecoderBlock(in_channels=embed_dim * 2, skip_channels=embed_dim, out_channels=embed_dim)
        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
    
    def forward(self, x):
        # Patch embedding.
        print('B: ',x.shape)
        # Assume input_tensor shape is [23, 1, 40, 128, 128]
        N, C, D, H, W = x.shape  # N=23, C=1, D=40, H=128, W=128
        # Reshape so that each slice becomes a separate sample
        x_reshaped = x.squeeze(1)  # New shape: [23*40, 1, 128, 128]
        x, H0, W0 = self.patch_embed(x_reshaped)  # x: (B, H0*W0, embed_dim)
        # Encoder stages with skip connections.
        x1, H1, W1 = self.layer1(x, H0, W0)  # stage1: (B, H1*W1, embed_dim)
        x2, H2, W2 = self.layer2(x1, H1, W1)  # stage2: (B, H2*W2, embed_dim*2)
        x3, H3, W3 = self.layer3(x2, H2, W2)  # stage3: (B, H3*W3, embed_dim*4)
        x4, H4, W4 = self.layer4(x3, H3, W3)  # stage4: (B, H4*W4, embed_dim*8)
        
        # Decoder.
        # Convert sequences back to spatial feature maps.
        x4_img = sequence_to_image(x4, H4, W4)  # (B, embed_dim*8, H4, W4)
        up3 = self.up3(x4_img)                   # Upsample to resolution of stage3.
        x3_img = sequence_to_image(x3, H3, W3)     # Skip connection from stage3.
        d3 = self.dec3(up3, x3_img)
        
        up2 = self.up2(d3)                       # Upsample to resolution of stage2.
        x2_img = sequence_to_image(x2, H2, W2)     # Skip connection from stage2.
        d2 = self.dec2(up2, x2_img)
        
        up1 = self.up1(d2)                       # Upsample to resolution of stage1.
        x1_img = sequence_to_image(x1, H1, W1)     # Skip connection from stage1.
        d1 = self.dec1(up1, x1_img)
        
        out = self.final_conv(d1)
        print('C: ', out.shape)
        out = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
        print('D: ', out.shape)
        # Optionally, upsample to original image size if needed.
        return out
    
    def load_pretrained(self, pretrained_path):
        """
        Loads pretrained weights.
        Note: The keys in the pretrained model may not exactly match if the architecture differs.
        Using strict=False to allow partial loading.
        """
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        self.load_state_dict(state_dict, strict=False)

##############################################
# Example usage
##############################################

# if __name__ == '__main__':
#     dummy_input = torch.randn(1, 3, 224, 224)
#     model = SwinUnet(img_size=224,
#                      patch_size=4,
#                      in_chans=3,
#                      num_classes=1,
#                      embed_dim=96,
#                      depths=[2, 2, 2, 2],
#                      num_heads=[3, 6, 12, 24],
#                      window_size=7,
#                      mlp_ratio=4.,
#                      dropout=0.)
#     output = model(dummy_input)
#     print("Output shape:", output.shape)
