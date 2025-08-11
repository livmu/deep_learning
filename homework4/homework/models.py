from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        h: int = 128
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.net = nn.Sequential(
            nn.Linear(n_track*4, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, n_waypoints*2)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]
        x = torch.cat([track_left, track_right], dim=1)
        
        out = self.net(x.view(B, -1))
        return out.view(B, self.n_waypoints, 2)
        

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: int = 0.1,
        num_layers: int = 4
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.pos_embed = nn.Embedding(2 * n_track, d_model)
        #self.pos_embed = nn.Parameter(0.01 * torch.randn(2 * n_track, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout=dropout,
            batch_first = True,
        )
        
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(2, d_model)                                         
        self.fc2 = nn.Linear(d_model, 2)

        input_mean = torch.tensor(INPUT_MEAN[:2], dtype=torch.float32)
        input_std = torch.tensor(INPUT_STD[:2], dtype=torch.float32)

        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]

        #mid = (track_left + track_right) / 2
        #track_left -= mid
        #track_right -= mid

        track_left = (track_left - self.input_mean[None, None, :]) / self.input_std[None, None, :]
        track_right = (track_right - self.input_mean[None, None, :]) / self.input_std[None, None, :]
        #track_left = track_left.view(B, -1)  # (b, n_track * 2)
        #track_right = track_right.view(B, -1)  # (b, n_track * 2)
        
        memory = torch.cat([track_left, track_right], dim=1)
        #memory = self.fc1(memory) + self.pos_embed.unsqueeze(0)
        memory = self.fc1(memory)
        memory += self.pos_embed(torch.arange(memory.size(1), device=memory.device)).unsqueeze(0).expand(B, -1, -1)
        
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.transformer(tgt=tgt, memory=memory)
        x = self.fc2(x)
        
        return x


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        h: int = 16,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        self.net = nn.Sequential(
            nn.Conv2d(n_waypoints, h, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(),
            nn.Conv2d(h, h*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(h*2),
            nn.ReLU(),
            nn.Conv2d(h*2, h*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(h*4),
            nn.ReLU(),
        )

        self.pool = nn.Sequential(
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(h*2, n_waypoints*2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        x = self.net(x)
        x = self.pool(x)
        x = x.view(-1, self.n_waypoints, 2)
        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
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
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
