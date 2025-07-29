import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    **kwargs,
):
    """
    Example detection (segmentation+depth) training function that replicates your style of train.py.
    You can call it like:

        from train_detection import train
        train(model_name="detector", num_epoch=10, lr=1e-3)

    It will:
      - load the drive dataset from 'drive_data/'
      - train a segmentation+depth model
      - log train/val segmentation IOU, accuracy, depth errors, etc.
      - save the model to homework/detector.th (for grading)
      - also save a copy in the timestamped log directory
    """

    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory for logs
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model
    model = load_model(model_name=model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load training and validation data
    train_data = load_data(
        dataset_path="drive_data/train",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_data = load_data(
        dataset_path="drive_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    # two losses
    seg_criterion = torch.nn.CrossEntropyLoss()
    depth_criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = DetectionMetric(num_classes=3)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        metric.reset()

        train_loss_vals = []

        # training loop
        for batch in train_data:
            x = batch["image"].to(device)       # (B, 3, H, W)
            seg_label = batch["track"].to(device)   # (B, H, W)
            depth_label = batch["depth"].to(device) # (B, H, W)

            optimizer.zero_grad()
            logits, raw_depth = model(x)

            seg_loss = seg_criterion(logits, seg_label)
            depth_loss = depth_criterion(raw_depth, depth_label)
            loss = seg_loss + depth_loss

            loss.backward()
            optimizer.step()

            train_loss_vals.append(loss.item())

            # track metrics
            with torch.inference_mode():
                pred_seg = logits.argmax(dim=1)
                metric.add(pred_seg, seg_label, raw_depth, depth_label)

            global_step += 1

        # gather training metrics
        train_metrics = metric.compute()
        train_iou = train_metrics["iou"]
        train_acc = train_metrics["accuracy"]
        train_depth_error = train_metrics["abs_depth_error"]
        train_tp_depth_err = train_metrics["tp_depth_error"]
        train_loss = np.mean(train_loss_vals)

        # validation
        model.eval()
        metric.reset()
        val_loss_vals = []

        with torch.inference_mode():
            for batch in val_data:
                x = batch["image"].to(device)
                seg_label = batch["track"].to(device)
                depth_label = batch["depth"].to(device)

                logits, raw_depth = model(x)
                seg_loss = seg_criterion(logits, seg_label)
                depth_loss = depth_criterion(raw_depth, depth_label)
                loss = seg_loss + depth_loss
                val_loss_vals.append(loss.item())

                pred_seg = logits.argmax(dim=1)
                metric.add(pred_seg, seg_label, raw_depth, depth_label)

        val_metrics = metric.compute()
        val_iou = val_metrics["iou"]
        val_acc = val_metrics["accuracy"]
        val_depth_error = val_metrics["abs_depth_error"]
        val_tp_depth_err = val_metrics["tp_depth_error"]
        val_loss = np.mean(val_loss_vals)

        # log to tensorboard
        logger.add_scalar("train_loss", train_loss, global_step)
        logger.add_scalar("train_iou", train_iou, global_step)
        logger.add_scalar("train_accuracy", train_acc, global_step)
        logger.add_scalar("train_abs_depth_error", train_depth_error, global_step)
        logger.add_scalar("train_tp_depth_error", train_tp_depth_err, global_step)

        logger.add_scalar("val_loss", val_loss, global_step)
        logger.add_scalar("val_iou", val_iou, global_step)
        logger.add_scalar("val_accuracy", val_acc, global_step)
        logger.add_scalar("val_abs_depth_error", val_depth_error, global_step)
        logger.add_scalar("val_tp_depth_error", val_tp_depth_err, global_step)

        # print on first, last, every 5 or 10 epochs
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d}: "
                f"train_loss={train_loss:.4f}, train_iou={train_iou:.3f}, "
                f"train_acc={train_acc:.3f}, train_depth_err={train_depth_error:.3f}, "
                f"train_tp_depth_err={train_tp_depth_err:.3f} || "
                f"val_loss={val_loss:.4f}, val_iou={val_iou:.3f}, "
                f"val_acc={val_acc:.3f}, val_depth_err={val_depth_error:.3f}, "
                f"val_tp_depth_err={val_tp_depth_err:.3f}"
            )

    # save model
    save_model(model)

    # also save to logging folder
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    # add other optional arguments (e.g., model hyperparams) as needed

    args = parser.parse_args()
    train(**vars(parser.parse_args()))