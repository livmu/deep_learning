import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb

from .metrics import DetectionMetric, ConfusionMatrix
from .models import Detector, load_model, save_model
from homework.datasets.road_dataset import load_data

def soft_iou(preds, targets, num_classes, eps=1e-6):
    preds = F.softmax(preds, dim=1)  # shape: (B, C, H, W) for segmentation, (B, C) for classification
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 2, 1).float()  # shape: (B, C, *)
    
    intersection = (preds * targets_onehot).sum(dim=0)
    union = (preds + targets_onehot - preds * targets_onehot).sum(dim=0)
    iou = (intersection + eps) / (union + eps)
    
    return 1 - iou.mean()


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
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

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()
    
    train_data = load_data("drive_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)
    #train_data, val_data = load_data(dataset_path='')

    # create loss function and optimizer
    train_metric = DetectionMetric(num_classes=3)
    val_metric = DetectionMetric(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = torch.tensor([0.2, 0.8, 1.0], device=device)
    track_criterion = torch.nn.CrossEntropyLoss()
    depth_criterion = torch.nn.L1Loss()
    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_metric.reset()
        val_metric.reset()
        model.train()
        train_loss_vals = []

        for x in train_data:
            img = x['image'].to(device)
            track = x['track'].to(device)
            depth = x['depth'].to(device)
            
            # TODO: implement training step
            optimizer.zero_grad()
            logits, raw_depth = model(img)

            #track = F.interpolate(track.unsqueeze(1).float(), size=logits.shape[-2:]).squeeze(1).long()
            #depth = F.interpolate(depth.unsqueeze(1), size=raw_depth.shape[-2:]).squeeze(1)
            
            #track_loss = track_criterion(logits, track)
            track_loss = 0.7 * F.cross_entropy(logits, track) + 0.3 * soft_iou(logits, track, num_classes)
            depth_loss = depth_criterion(raw_depth, depth)
            loss = track_loss + depth_loss
            loss.backward()
            optimizer.step()

            train_loss_vals.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_metric.add(preds, track, raw_depth, depth)

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1
        

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            val_loss_vals = []

            for x in val_data:
                img = x['image'].to(device)
                track = x['track'].to(device)
                depth = x['depth'].to(device)
        
                # TODO: compute validation accuracy
                logits, raw_depth = model(img)
                track_loss = track_criterion(logits, track)
                depth_loss = depth_criterion(raw_depth, depth)
                loss = track_loss + depth_loss
                val_loss_vals.append(loss.item())
                
                preds = torch.argmax(logits, dim=1)
                val_metric.add(preds, track, raw_depth, depth)

        # log average train and val accuracy to tensorboard
        train = train_metric.compute()
        train_acc = train['accuracy']
        train_iou = train['iou']
        train_err = train['abs_depth_error']
        train_tp_err = train['tp_depth_error']
        train_loss = np.mean(train_loss_vals)
        
        val = val_metric.compute()
        val_acc = val['accuracy']
        val_iou = val['iou']
        val_err = val['abs_depth_error']
        val_tp_err = val['tp_depth_error']
        val_loss = np.mean(val_loss_vals)

        logger.add_scalar("train_acc", train_acc, global_step)
        logger.add_scalar("train_iou", train_iou, global_step)
        logger.add_scalar("train_err", train_err, global_step)
        logger.add_scalar("train_tp_err", train_tp_err, global_step)
        logger.add_scalar("train_loss", train_loss, global_step)

        logger.add_scalar("val_acc", val_acc, global_step)
        logger.add_scalar("val_iou", val_iou, global_step)
        logger.add_scalar("val_err", val_err, global_step)
        logger.add_scalar("val_tp_err", val_tp_err, global_step)
        logger.add_scalar("val_loss", val_loss, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_acc:.4f} "
                f"train_iou={train_iou:.4f} "
                f"train_err={train_err:.4f} "
                f"train_tp_err={train_tp_err:.4f} "
                f"train_loss={train_loss:.4f} "
                f"val_acc={val_acc:.4f}"
                f"val_iou={val_iou:.4f} "
                f"val_err={val_err:.4f} "
                f"val_tp_err={val_tp_err:.4f} "
                f"val_loss={val_loss:.4f} "
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
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
