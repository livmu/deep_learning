import argparse
import time as time_module
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric



def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Start timing
    start_time = time_module.time()

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Create model - using Detector for road detection
    model = Detector(in_channels=3, num_classes=3, **kwargs)
    model = model.to(device)
    model.train()

    # Load data - using road dataset
    train_data = load_data("drive_data/train", 
                          transform_pipeline=transform_pipeline,
                          shuffle=True, 
                          batch_size=batch_size, 
                          num_workers=2)
    val_data = load_data("drive_data/val", 
                        transform_pipeline="default",  # No augmentation for validation
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=2)

    # create loss functions and optimizer as specified in README
    # Cross-entropy loss for segmentation with more balanced class weights
    # Less extreme weights to avoid overfitting to minority classes
    class_weights = torch.tensor([0.8, 10.0, 10.0]).to(device)  # Reduce extreme weighting
    seg_loss_func = nn.CrossEntropyLoss(weight=class_weights)
    
    # Regression loss for depth prediction (L1Loss/MAE works better than MSE for outliers)
    depth_loss_func = nn.L1Loss()  # Mean Absolute Error - a regression loss as specified
    
    # Use AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Add learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    global_step = 0
    
    # Use detection metrics (mIoU for segmentation, MAE for depth)
    train_detection_metric = DetectionMetric(num_classes=3)
    val_detection_metric = DetectionMetric(num_classes=3)
    
    best_val_iou = 0.0

    # training loop
    for epoch in range(num_epoch):
        # Reset metrics at beginning of epoch
        train_detection_metric.reset()
        val_detection_metric.reset()

        model.train()
        epoch_train_seg_loss = 0.0
        epoch_train_depth_loss = 0.0
        epoch_train_total_loss = 0.0
        num_train_batches = 0

        for batch in train_data:
            # Extract data from batch
            img = batch["image"].to(device)
            track_mask = batch["track"].to(device)  # segmentation labels (b, h, w)
            depth_gt = batch["depth"].to(device)    # depth ground truth (b, h, w)

            # Training step
            optimizer.zero_grad()  # clear gradients
            logits, depth_pred = model(img)  # forward pass
            
            # Compute losses with heavy emphasis on segmentation
            seg_loss = seg_loss_func(logits, track_mask)
            depth_loss = depth_loss_func(depth_pred, depth_gt)
            
            # More balanced loss weighting
            total_loss = 3.0 * seg_loss + depth_loss
            
            # Log individual losses
            logger.add_scalar("train_seg_loss", seg_loss.item(), global_step=global_step)
            logger.add_scalar("train_depth_loss", depth_loss.item(), global_step=global_step)
            logger.add_scalar("train_total_loss", total_loss.item(), global_step=global_step)
            
            epoch_train_seg_loss += seg_loss.item()
            epoch_train_depth_loss += depth_loss.item()
            epoch_train_total_loss += total_loss.item()
            num_train_batches += 1

            # Calculate training metrics using model's predict method
            seg_pred, depth_pred_final = model.predict(img)
            train_detection_metric.add(seg_pred, track_mask, depth_pred_final, depth_gt)

            total_loss.backward()  # backward pass
            optimizer.step()  # update weights

            global_step += 1

        # Validation phase
        with torch.inference_mode():
            model.eval()
            epoch_val_seg_loss = 0.0
            epoch_val_depth_loss = 0.0
            epoch_val_total_loss = 0.0
            num_val_batches = 0

            for batch in val_data:
                # Extract data from batch
                img = batch["image"].to(device)
                track_mask = batch["track"].to(device)
                depth_gt = batch["depth"].to(device)

                # Validation step
                logits, depth_pred = model(img)
                seg_loss = seg_loss_func(logits, track_mask)
                depth_loss = depth_loss_func(depth_pred, depth_gt)
                total_loss = 5.0 * seg_loss + depth_loss
                
                epoch_val_seg_loss += seg_loss.item()
                epoch_val_depth_loss += depth_loss.item()
                epoch_val_total_loss += total_loss.item()
                num_val_batches += 1
                
                # Calculate validation metrics
                seg_pred, depth_pred_final = model.predict(img)
                val_detection_metric.add(seg_pred, track_mask, depth_pred_final, depth_gt)

        # Compute epoch metrics
        train_metrics = train_detection_metric.compute()
        val_metrics = val_detection_metric.compute()
        
        epoch_train_iou = train_metrics["iou"]
        epoch_train_acc = train_metrics["accuracy"]
        epoch_train_depth_error = train_metrics["abs_depth_error"]
        
        epoch_val_iou = val_metrics["iou"]
        epoch_val_acc = val_metrics["accuracy"]
        epoch_val_depth_error = val_metrics["abs_depth_error"]
        
        avg_train_seg_loss = epoch_train_seg_loss / num_train_batches
        avg_train_depth_loss = epoch_train_depth_loss / num_train_batches
        avg_train_total_loss = epoch_train_total_loss / num_train_batches
        
        avg_val_seg_loss = epoch_val_seg_loss / num_val_batches
        avg_val_depth_loss = epoch_val_depth_loss / num_val_batches
        avg_val_total_loss = epoch_val_total_loss / num_val_batches

        # Log epoch metrics
        logger.add_scalar("epoch_train_iou", epoch_train_iou, global_step=epoch)
        logger.add_scalar("epoch_train_acc", epoch_train_acc, global_step=epoch)
        logger.add_scalar("epoch_train_depth_error", epoch_train_depth_error, global_step=epoch)
        logger.add_scalar("epoch_train_seg_loss", avg_train_seg_loss, global_step=epoch)
        logger.add_scalar("epoch_train_depth_loss", avg_train_depth_loss, global_step=epoch)
        logger.add_scalar("epoch_train_total_loss", avg_train_total_loss, global_step=epoch)
        
        logger.add_scalar("epoch_val_iou", epoch_val_iou, global_step=epoch)
        logger.add_scalar("epoch_val_acc", epoch_val_acc, global_step=epoch)
        logger.add_scalar("epoch_val_depth_error", epoch_val_depth_error, global_step=epoch)
        logger.add_scalar("epoch_val_seg_loss", avg_val_seg_loss, global_step=epoch)
        logger.add_scalar("epoch_val_depth_loss", avg_val_depth_loss, global_step=epoch)
        logger.add_scalar("epoch_val_total_loss", avg_val_total_loss, global_step=epoch)

        # Save best model based on validation IoU
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            save_model(model)
            print(f"New best model saved! Val IoU: {epoch_val_iou:.4f}")
        
        # Step the learning rate scheduler
        scheduler.step(epoch_val_iou)

        # Print progress every epoch
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={avg_train_total_loss:.4f} "
            f"train_iou={epoch_train_iou:.4f} "
            f"train_depth_err={epoch_train_depth_error:.4f} "
            f"val_loss={avg_val_total_loss:.4f} "
            f"val_iou={epoch_val_iou:.4f} "
            f"val_depth_err={epoch_val_depth_error:.4f}"
        )

    end_time = time_module.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed for {model_name}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    
    # Log final metrics
    logger.add_scalar("training/total_time_seconds", training_time, global_step=0)
    logger.add_scalar("training/time_per_epoch", training_time/num_epoch, global_step=0)
    logger.add_scalar("training/best_val_iou", best_val_iou, global_step=0)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default", 
                        help="Transform pipeline: 'default' or 'aug' for augmentation")

    # pass all arguments to train
    train(**vars(parser.parse_args()))