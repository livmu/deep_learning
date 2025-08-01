import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .metrics import AccuracyMetric#, DetectionMetric, ConfusionMatrix
from .models import Classifier, load_model, save_model
from homework.datasets.classification_dataset import load_data
#from datasets.road_dataset import load_data
#from datasets.road_transforms import load_data
#from datasets.road_utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
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
    
    # change
    train_data = load_data("classification_data/train", transform_pipeline='aug', shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("classification_data/val", shuffle=False)
    #train_data, val_data = load_data(dataset_path='')

    # create loss function and optimizer
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            optimizer.zero_grad()
            logits = model(img)
            loss = torch.nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_metric.add(preds, label)

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)
        
                # TODO: compute validation accuracy
                logits = model(img)
                preds = torch.argmax(logits, dim=1)
                val_metric.add(preds, label)

        # log average train and val accuracy to tensorboard
        train_acc = train_metric.compute()['accuracy']
        val_acc = val_metric.compute()['accuracy']

        logger.add_scalar("train_acc", train_acc, global_step)
        logger.add_scalar("val_acc", val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f}"
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
