from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        layer1: int = 32,
        layer2: int = 64,
        layer3: int = 128,
        s1: int = 8,
        s2: int = 2,
        s3: int = 1
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
        self.conv1 = nn.Conv2d(in_channels, layer1, kernel_size=3, stride=s1, padding=1)
        self.conv2 = nn.Conv2d(layer1, layer2, kernel_size=3, stride=s2, padding=1)
        self.conv3 = nn.Conv2d(layer2, layer3, kernel_size=3, stride=s3, padding=1)
        #self.conv4 = nn.Conv2d(layer3, num_classes, kernel_size=3, stride=s3, padding=1)

        self.batch1 = nn.BatchNorm2d(layer1)
        self.batch2 = nn.BatchNorm2d(layer2)
        self.batch3 = nn.BatchNorm2d(layer3)
        
        #self.fc1 = nn.Linear(layer3 * 8 * 8, 256)
        self.fc1 = nn.Linear(layer3, num_classes)
        
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        z = self.batch1(self.conv1(z))
        z = self.max_pool(self.relu(z))
        
        z = self.batch2(self.conv2(z))
        z = self.max_pool(self.relu(z))
        
        z = self.batch3(self.conv3(z))
        z = self.avg_pool(self.relu(z))

        logits = self.fc1(z.flatten(1))
        #z = self.dropout(self.avg_pool(z))
        #logits = self.conv4(z).squeeze(-1).squeeze(-1)
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


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        layer1: int = 16,
        layer2: int = 32,
        k: int = 3,
        s: int = 1,
        p: int = 1
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
        
        # TODO: implement
        self.conv1 = nn.Conv2d(in_channels, layer1, kernel_size=k, stride=s, padding=p)
        self.batch1 = nn.BatchNorm2d(layer1)
        
        '''self.down = nn.Sequential(
            nn.Conv2d(layer1, layer1, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(layer1),
            nn.ReLU(inplace=True),
            nn.Conv2d(layer1, layer1, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(layer1),
            nn.ReLU(inplace=True),
        )'''

        self.conv2 = nn.Conv2d(layer1, layer2, kernel_size=k, stride=s, padding=p)
        self.batch2 = nn.BatchNorm2d(layer2)

        '''self.up = nn.Sequential(
            nn.ConvTranspose2d(layer2, layer2, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(layer2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(layer2, layer2, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(layer2),
            nn.ReLU(inplace=True),
        )'''

        self.conv3 = nn.ConvTranspose2d(layer2, layer1, kernel_size=4, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(layer1)

        self.conv4 = nn.ConvTranspose2d(layer1, layer1, kernel_size=4, stride=2, padding=1)
        self.batch4 = nn.BatchNorm2d(layer1)

        self.track_head = nn.Conv2d(layer1, num_classes, kernel_size=k, stride=s, padding=p)
        self.depth_head = nn.Conv2d(layer1, 1, kernel_size=k, stride=s, padding=p)
        #self.pool = nn.AdaptiveAvgPool2d((1,1))

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

        # TODO: replace with actual forward pass
        z = self.batch1(self.conv1(z))
        #z = self.down(z)

        z = self.batch2(self.conv2(z))
        #z = self.up(z)

        z = self.batch3(self.conv3(z))
        z = self.batch4(self.conv4(z))

        logits = self.track_head(z)
        raw_depth = self.depth_head(z).squeeze(1)
        raw_depth = F.interpolate(raw_depth.unsqueeze(1), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False).squeeze(1)

        return logits, raw_depth

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
        depth = torch.clamp(raw_depth, 0.0, 1.0)

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
