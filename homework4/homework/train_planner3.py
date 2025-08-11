"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
import numpy as np
from sympy.physics.units import momentum
from torch import nn
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import load_data

# data visualization imports
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tkinter import Tk, Canvas, Button, filedialog, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from homework.metrics import PlannerMetric
from homework.models import MLPPlanner, save_model, TransformerPlanner, CNNPlanner


def visualize_sample(sample, canvas_frame):

    # visualize image
    # get picture of the sample
    image = sample["image"]
    image *= 255

    # create uint8 array and transpose to (96, 128, 3)
    image = np.uint8(image)
    image = image.transpose(1, 2, 0)
    # image is an ndarray with shape (3, 96, 128) and values in [0, 1] range and type float32 convert to PIL image and display
    image = transforms.ToPILImage()(image)


    # visualize track

    track_left = sample["track_left"]  # Shape: (n_track, 2)
    track_right = sample["track_right"]  # Shape: (n_track, 2)
    waypoints = sample["waypoints"]  # Shape: (n_waypoints, 2)
    waypoints_mask = sample["waypoints_mask"]  # Shape: (n_waypoints,)

    # shift the track left and right and waypoints by 12 units
    track_left[:, 0] += 12
    track_right[:, 0] += 12
    waypoints[:, 0] += 12


    plt.figure(figsize=(12, 10))


    # Plot track boundaries
    plt.plot(track_left[:, 0], track_left[:, 1], 'r-', label="Left Lane Boundary")
    plt.plot(track_right[:, 0], track_right[:, 1], 'b-', label="Right Lane Boundary")

    # Plot target waypoints
    clean_waypoints = waypoints[waypoints_mask]


    # set plt static limits
    plt.xlim(0, 24)
    plt.ylim(0, 25)


    plt.scatter(clean_waypoints[:, 0], clean_waypoints[:, 1], c='g', label="Target Waypoints")

    # overlay the image on the plot
    plt.imshow(image, extent=[0, 24, 0, 25])


    # Add labels and legend
    plt.title("Sample Visualization")
    plt.xlabel("X-axis (meters)")
    plt.ylabel("Y-axis (meters)")
    plt.legend()
    plt.grid(True)

    # Draw the matplotlib figure on the tkinter canvas
    # remove the previous canvas
    for widget in canvas_frame.winfo_children():
        widget.destroy()
    canvas_tkagg = FigureCanvasTkAgg(plt.gcf(), canvas_frame)
    canvas_tkagg.get_tk_widget().pack(fill="both", expand=True)
    canvas_tkagg.draw()


def create_ui(data_loader):
    """Creates a Tkinter UI for visualizing dataset samples."""
    root = Tk()
    root.title("Dataset Visualizer")

    # Frame to hold the canvas
    canvas_frame = Frame(root)
    canvas_frame.pack(side="top", fill="both", expand=True)

    # Placeholder canvas for matplotlib figures
    canvas = Canvas(canvas_frame)
    canvas.pack(fill="both", expand=True)

    # show all the samples in the dataset
    for i in range(len(data_loader.dataset)):
        sample = data_loader.dataset[i]
        visualize_sample(sample, canvas_frame)
        root.update()

    root.mainloop()


def validate_with_metrics(model, val_loader, device):
    """
    Computes validation metrics using PlannerMetric.

    Args:
        model (nn.Module): The model being evaluated.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run validation on (CPU, GPU, or MPS).

    Returns:
        dict: Dictionary of validation metrics including longitudinal, lateral, and L1 errors.
    """
    model.eval()
    metric = PlannerMetric()
    metric.reset()

    with torch.no_grad():  # Disable gradient calculations for validation
        for batch in val_loader:
            # Move data to the appropriate device
            track_left = batch["track_left"].to(device)  # Shape: (b, n_track, 2)
            track_right = batch["track_right"].to(device)  # Shape: (b, n_track, 2)
            waypoints = batch["waypoints"].to(device)  # Shape: (b, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # Shape: (b, n_waypoints)

            # Forward pass
            predicted_waypoints = model(track_left, track_right)  # Shape: (b, n_waypoints, 2)

            # Update the metric
            metric.add(preds=predicted_waypoints, labels=waypoints, labels_mask=waypoints_mask)

    # Compute the final metrics
    metrics = metric.compute()
    #print(f"Validation Metrics: {metrics}")
    return metrics


def train():
    """
    Train the MLPPlanner model.
    """
    # Set random seed for reproducibility
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device setup
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Data loading
    batch_size = 64
    train_data: DataLoader = load_data(
        "drive_data/train", shuffle=True, batch_size=batch_size, num_workers=4, transform_pipeline="state_only"
    )
    val_data: DataLoader = load_data(
        "drive_data/val", shuffle=False, batch_size=batch_size, transform_pipeline="default"
    )


    # Model initialization
    model = MLPPlanner().to(device)
    print(model)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    # Gradient clipping parameters
    max_grad_norm = 1.0  # Maximum gradient norm

    # Metrics initialization
    planner_metric = PlannerMetric()

    num_epochs = 100
    lat_err = float("inf")  # Initialize the best lateral error

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        planner_metric.reset()

        for batch in train_data:
            # Load batch data
            track_left = batch["track_left"].to(device)  # (b, n_track, 2)
            track_right = batch["track_right"].to(device)  # (b, n_track, 2)
            waypoints = batch["waypoints"].to(device)  # (b, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (b, n_waypoints)

            # Forward pass
            predicted_waypoints = model(track_left, track_right)  # (b, n_waypoints, 2)

            # Fix mask shape to match the target tensor
            valid_waypoints_mask = waypoints_mask.unsqueeze(-1).expand(-1, -1, 2)  # (b, n_waypoints, 2)

            # Use mask to select valid elements
            masked_predicted = predicted_waypoints[valid_waypoints_mask]
            masked_targets = waypoints[valid_waypoints_mask]

            # Compute loss
            loss = criterion(masked_predicted, masked_targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            num_batches += 1
            planner_metric.add(predicted_waypoints, waypoints, waypoints_mask)

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_metrics = planner_metric.compute()

        # Validation phase
        val_metrics = validate_with_metrics(model, val_data, device)

        # Save the model if the validation lateral error improves
        if val_metrics['lateral_error'] < lat_err:
            lat_err = val_metrics['lateral_error']
            save_model(model)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, "
            f"L1 Error: {train_metrics['l1_error']:.6f}, Longitudinal Error: {train_metrics['longitudinal_error']:.6f}, "
            f"Lateral Error: {train_metrics['lateral_error']:.6f}, Validation L1 Error: {val_metrics['l1_error']:.6f}"
        )


def train_transformer():
    """
    Train the TransformerPlanner model.
    """
    # Set random seed for reproducibility
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device setup
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Data loading
    batch_size = 64
    train_data: DataLoader = load_data(
        "../drive_data/train", shuffle=True, batch_size=batch_size, num_workers=4, transform_pipeline="default"
    )
    val_data: DataLoader = load_data(
        "../drive_data/val", shuffle=False, batch_size=batch_size, transform_pipeline="default"
    )

    # Model initialization
    model = TransformerPlanner().to(device)
    print(model)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4
    )
    criterion = nn.SmoothL1Loss()

    # Gradient clipping parameters
    max_grad_norm = 1.0  # Maximum gradient norm

    # Metrics initialization
    planner_metric = PlannerMetric()

    num_epochs = 100
    lat_err = float("inf")  # Initialize the best lateral error

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        planner_metric.reset()

        for batch in train_data:
            # Load batch data
            track_left = batch["track_left"].to(device)  # (b, n_track, 2)
            track_right = batch["track_right"].to(device)  # (b, n_track, 2)
            waypoints = batch["waypoints"].to(device)  # (b, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (b, n_waypoints)

            # Forward pass
            predicted_waypoints = model(track_left, track_right)  # (b, n_waypoints, 2)

            # Fix mask shape to match the target tensor
            valid_waypoints_mask = waypoints_mask.unsqueeze(-1).expand(-1, -1, 2)  # (b, n_waypoints, 2)

            # Use mask to select valid elements
            masked_predicted = predicted_waypoints[valid_waypoints_mask]
            masked_targets = waypoints[valid_waypoints_mask]

            # Compute loss
            loss = criterion(masked_predicted, masked_targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            num_batches += 1
            planner_metric.add(predicted_waypoints, waypoints, waypoints_mask)

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_metrics = planner_metric.compute()

        # Validation phase
        val_metrics = validate_with_metrics(model, val_data, device)

        # Step the scheduler based on longitudinal error
        scheduler.step(val_metrics['longitudinal_error'])

        # Save the model if the validation lateral error improves
        if val_metrics['lateral_error'] < lat_err:
            lat_err = val_metrics['lateral_error']
            save_model(model)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, "
            f"L1 Error: {train_metrics['l1_error']:.6f}, Longitudinal Error: {train_metrics['longitudinal_error']:.6f}, "
            f"Lateral Error: {train_metrics['lateral_error']:.6f}, Validation L1 Error: {val_metrics['l1_error']:.6f}"
        )


def validate_with_metrics_cnn(model, data_loader, device):
    """
    Validates the model on the validation dataset and computes metrics.

    Args:
        model: The model to validate.
        data_loader: DataLoader for the validation dataset.
        device: The device to run validation on.

    Returns:
        dict: Validation metrics.
    """
    model.eval()
    planner_metric = PlannerMetric()
    planner_metric.reset()

    with torch.no_grad():
        for batch in data_loader:
            # Load batch data
            images = batch["image"].to(device)  # (b, 3, h, w)
            waypoints = batch["waypoints"].to(device)  # (b, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (b, n_waypoints)

            # Forward pass
            predicted_waypoints = model(images)  # (b, n_waypoints, 2)

            # Add metrics
            planner_metric.add(predicted_waypoints, waypoints, waypoints_mask)

    return planner_metric.compute()


def train_cnn():
    """
    Train the CNNPlanner model.
    """
    # Set random seed for reproducibility
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device setup
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    # Data loading
    batch_size = 64
    train_data: DataLoader = load_data(
        "../drive_data/train", shuffle=True, batch_size=batch_size, num_workers=4, transform_pipeline="aug"
    )
    val_data: DataLoader = load_data(
        "../drive_data/val", shuffle=False, batch_size=batch_size, transform_pipeline="default"
    )

    # Model initialization
    model = CNNPlanner(n_waypoints=3).to(device)
    print(model)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=1e-4
    )
    criterion = nn.SmoothL1Loss()

    # Gradient clipping parameters
    max_grad_norm = 1.0  # Maximum gradient norm

    # Metrics initialization
    planner_metric = PlannerMetric()

    num_epochs = 100
    lat_err = float("inf")  # Initialize the best lateral error

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        planner_metric.reset()

        for batch in train_data:
            # Load batch data
            images = batch["image"].to(device)  # (b, 3, 96, 128)
            waypoints = batch["waypoints"].to(device)  # (b, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (b, n_waypoints)

            # Forward pass
            predicted_waypoints = model(images)  # (b, n_waypoints, 2)

            # Fix mask shape to match the target tensor
            valid_waypoints_mask = waypoints_mask.unsqueeze(-1).expand(-1, -1, 2)  # (b, n_waypoints, 2)

            # Use mask to select valid elements
            masked_predicted = predicted_waypoints[valid_waypoints_mask]
            masked_targets = waypoints[valid_waypoints_mask]

            # Compute loss
            loss = criterion(masked_predicted, masked_targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            num_batches += 1
            planner_metric.add(predicted_waypoints, waypoints, waypoints_mask)

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_metrics = planner_metric.compute()

        # Validation phase
        val_metrics = validate_with_metrics_cnn(model, val_data, device)

        # Step the scheduler based on longitudinal error
        scheduler.step(val_metrics['longitudinal_error'])

        # Save the model if the validation lateral error improves
        if val_metrics['lateral_error'] < lat_err:
            lat_err = val_metrics['lateral_error']
            save_model(model)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, "
            f"L1 Error: {train_metrics['l1_error']:.6f}, Longitudinal Error: {train_metrics['longitudinal_error']:.6f}, "
            f"Lateral Error: {train_metrics['lateral_error']:.6f}, Validation L1 Error: {val_metrics['l1_error']:.6f}"
        )


if __name__ == "__main__":
    train()