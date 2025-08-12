"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt

from .metrics import PlannerMetric
from .models import load_model, save_model
from homework.datasets.road_dataset import load_data

def plot_waypoints(pred, target, idx=0, invert_y=False, title=None):
    """
    Plot predicted vs ground-truth waypoints for a single sample.

    pred   : torch.Tensor of shape (B, T, 2)
    target : torch.Tensor of shape (B, T, 2)
    idx    : index of batch item to plot
    invert_y : set True if you want lateral axis flipped
    title  : optional string for plot title
    """
    # Select one sample and move to CPU
    p = pred[idx].detach().cpu().numpy()
    t = target[idx].detach().cpu().numpy()

    # Optionally invert y axis (depends on your coord frame)
    if invert_y:
        p[:, 1] = -p[:, 1]
        t[:, 1] = -t[:, 1]

    plt.figure(figsize=(5, 5))
    plt.plot(t[:, 0], t[:, 1], 'o-', label='Ground Truth', color='green')
    plt.plot(p[:, 0], p[:, 1], 'x--', label='Predicted', color='red')
    plt.scatter(t[0, 0], t[0, 1], c='blue', marker='s', label='Start')

    plt.xlabel("Longitudinal (m)")
    plt.ylabel("Lateral (m)")
    plt.legend()
    plt.grid(True)
    if title:
        plt.title(title)
    plt.axis('equal')  # Keep aspect ratio equal for accurate geometry
    plt.show()

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    transform_pipeline: str = "state_only",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 4,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final 
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load data
    train_data = load_data(
        "drive_data/train", 
        transform_pipeline=transform_pipeline, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers,
    )
    
    val_data = load_data("drive_data/val", shuffle=False)
    #train_data, val_data = load_data(dataset_path='')

    all_wps = []
    for batch in train_data:
        wps = batch["waypoints"].view(-1, 2)  # flatten (B, T, 2) -> (B*T, 2)
        all_wps.append(wps)
    all_wps = torch.cat(all_wps, dim=0)
    wp_mean = all_wps.mean(dim=0, keepdim=True).to(device)  # (1, 2)
    wp_std = all_wps.std(dim=0, keepdim=True).to(device) + 1e-8

    # create loss function and optimizer
    # weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #weights = torch.tensor([0.2, 0.8, 1.0], device=device)
    if model_name == "mlp_planner": criterion = nn.MSELoss()
    else: criterion = F.smooth_l1_loss

    global_step = 0
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()

        train_loss = 0.0
        val_loss = 0.0
        train_count = 0
        val_count = 0
        
        model.train()

        for batch in train_data:
            if model_name == "cnn_planner":
                img = batch.get("image").to(device)
                logits = model(img)
            else:
                track_left = batch.get("track_left").to(device)
                track_right = batch.get("track_right").to(device)
                logits = model(track_left, track_right)
                
            waypoints = batch.get("waypoints").to(device)
            waypoints_mask = batch.get("waypoints_mask").to(device)

            wps_norm = (waypoints - wp_mean) / wp_std
            
            # TODO: implement training 
            optimizer.zero_grad()

            loss = criterion(logits, waypoints)
            #loss = criterion(logits[waypoints_mask], waypoints[waypoints_mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds_denorm = preds_norm * wp_std + wp_mean
            train_metric.add(preds_denorm, wps, mask)

            #preds = torch.argmax(logits, dim=1)
            train_metric.add(logits, waypoints, waypoints_mask)
            train_loss += loss.item()
            train_count += 1
            
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                if model_name == "cnn_planner":
                    img = batch.get("image").to(device)
                    logits = model(img)
                else:
                    track_left = batch.get("track_left").to(device)
                    track_right = batch.get("track_right").to(device)
                    logits = model(track_left, track_right)
                
                waypoints = batch.get("waypoints").to(device)
                waypoints_mask = batch.get("waypoints_mask").to(device)
        
                # TODO: compute validation accuracy
                val_metric.add(logits, waypoints, waypoints_mask)

                #val_count += 1
                #if val_count == 1:
                #    plot_waypoints(logits, waypoints, idx=0, invert_y=False, title="Pred vs GT")
                #    break
                

        avg_train_loss = train_loss / train_count        
        train_result = train_metric.compute()
        val_result = val_metric.compute()

        # print on first, last, every 10th epoch
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss: {avg_train_loss:.4f} | "
            f"train_long_err: {train_result['longitudinal_error']:.4f} | "
            f"val_long_err: {val_result['longitudinal_error']:.4f} | "
            f"train_lat_err: {train_result['lateral_error']:.4f} | "
            f"val_lat_err: {val_result['lateral_error']:.4f} "
        )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
