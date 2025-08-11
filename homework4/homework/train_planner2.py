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
import torch.utils.tensorboard as tb

from .metrics import PlannerMetric
from .models import load_model, save_model
from homework.datasets.road_dataset import load_data


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

    # create loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #weights = torch.tensor([0.2, 0.8, 1.0], device=device)
    criterion = nn.MSELoss()

    global_step = 0
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metrics.reset()
        val_metrics.reset()
        
        model.train()

        for batch in train_data:
            track_left = batch.get("track_left").to(device)
            track_right = batch.get("track_right").to(device)
            waypoints = batch.get("waypoints").to(device)
            waypoints_mask = batch.get("waypoints_mask").to(device)

            # TODO: implement training 
            logits = model(track_left, track_right)
            optimizer.zero_grad()
            
            loss = criterion(logits, waypoints)
            loss.backward()
            optimizer.step()

            #preds = torch.argmax(logits, dim=1)
            train_metric.add(logits, waypoints, waypoints_mask)

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left = batch.get("track_left").to(device)
                track_right = batch.get("track_right").to(device)
                
                    
                waypoints = batch.get("waypoints").to(device)
                waypoints_mask = batch.get("waypoints_mask").to(device)
        
                # TODO: compute validation accuracy
                logits = model(track_left, track_right)
                val_metric.add(logits, waypoints, waypoints_mask)

        # log average train and val accuracy to tensorboard
        train_acc = train_metric.compute()['accuracy']
        val_acc = val_metric.compute()['accuracy']

        logger.add_scalar("train_acc", train_acc, global_step)
        logger.add_scalar("val_acc", val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {avg_val_loss:.4f} | "
                f"long_err: {results['longitudinal_error']:.4f} | "
                f"lat_err: {results['lateral_error']:.4f}"
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
