from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        # Example: a simple CNN architecture
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Input is 64x64, after 3 max pools (2x2, stride=2): 64/8 = 8
        # So final conv output is 128 channels * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Forward pass through conv layers
        z = self.pool(self.relu(self.conv1(z)))  # 64x64 -> 32x32
        z = self.pool(self.relu(self.conv2(z)))  # 32x32 -> 16x16  
        z = self.pool(self.relu(self.conv3(z)))  # 16x16 -> 8x8
        
        # Flatten for fully connected layers
        z = z.view(z.size(0), -1)  # (batch, 128*8*8)
        
        # Fully connected layers
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        logits = self.fc2(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class DownBlock(nn.Module):
    """Downsampling block with conv + batchnorm + relu + stride + residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection for residual
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # Add residual connection
        out = self.relu(out)
        
        return out


class UpBlock(nn.Module):
    """Upsampling block with transpose conv + batchnorm + relu + skip connections + residual"""
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # If skip connections, we need to handle concatenated features
        conv_in_channels = out_channels + skip_channels
        
        # Two conv blocks for residual connection
        self.conv1 = nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection for residual - may need projection if channels differ
        if conv_in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(conv_in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_proj = nn.Identity()
        
    def forward(self, x, skip=None):
        x = self.upconv(x)
        if skip is not None:
            # Use interpolation to match exact dimensions
            x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        # Store input for residual connection
        residual = self.residual_proj(x)
        
        # Two conv blocks with residual connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add residual connection
        out = self.relu(out + residual)
        
        return out


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Encoder (downsampling path) - simpler but effective
        self.down1 = DownBlock(in_channels, 32)      # (B, 3, H, W) -> (B, 32, H/2, W/2)
        self.down2 = DownBlock(32, 64)               # (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        
        # Simplified bottleneck to reduce computation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling path) with skip connections
        self.up1 = UpBlock(64, 32, skip_channels=64)     # (B, 64, H/4, W/4) + skip -> (B, 32, H/2, W/2)
        self.up2 = UpBlock(32, 32, skip_channels=32)     # (B, 32, H/2, W/2) + skip -> (B, 32, H, W)
        
        # Task-specific heads with final upsampling to ensure correct output size
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, num_classes, kernel_size=1)  # (B, 32, H, W) -> (B, 3, H, W)
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),         # (B, 32, H, W) -> (B, 1, H, W)
            nn.Sigmoid()                             # Constrain depth to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder (downsampling) - save features for skip connections
        down1_out = self.down1(z)          # (B, 3, H, W) -> (B, 32, H/2, W/2)
        down2_out = self.down2(down1_out)  # (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(down2_out)  # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        
        # Decoder (upsampling) with skip connections
        up1_out = self.up1(bottleneck_out, down2_out)  # (B, 64, H/4, W/4) + skip -> (B, 32, H/2, W/2)
        up2_out = self.up2(up1_out, down1_out)         # (B, 32, H/2, W/2) + skip -> (B, 32, H, W)
        
        # Task-specific heads
        logits = self.seg_head(up2_out)   # (B, 32, H/?, W/?) -> (B, 3, H/?, W/?)
        depth_raw = self.depth_head(up2_out)  # (B, 32, H/?, W/?) -> (B, 1, H/?, W/?)
        
        # Ensure output matches input size exactly
        target_size = (x.shape[2], x.shape[3])  # (H, W) from original input
        logits = torch.nn.functional.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        depth_raw = torch.nn.functional.interpolate(depth_raw, size=target_size, mode='bilinear', align_corners=False)
        depth = depth_raw.squeeze(1)      # (B, 1, H, W) -> (B, H, W)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()