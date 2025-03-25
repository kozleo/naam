from torchvision import datasets, transforms
import random
from PIL import Image
import torch
import numpy as np


def mask_random_squares(batch, p, s, device):
    """
    Masks p squares of size s for each image in a batch and returns a binary tensor indicating masked pixels.
    Creates a copy of the batch before modifying.

    Parameters:
    batch (torch.Tensor): A batch of images with shape (N, C, H, W).
    p (int): Number of squares to mask.
    s (int): Size of each square.

    Returns:
    torch.Tensor: The batch of images with masked squares.
    torch.Tensor: Binary tensor with ones at masked pixels and zeros elsewhere.
    """
    # Ensure the square size is not larger than the image size
    s = min(s, batch.shape[2], batch.shape[3])

    # Create a copy of the batch to modify
    modified_batch = batch.clone()

    # Initialize a binary tensor for masked pixels
    binary_mask = torch.zeros(batch.shape, device=device)

    # Generate square coordinates for the entire batch
    square_coords = [(random.randint(0, batch.shape[2] - s), random.randint(0, batch.shape[3] - s)) for _ in range(p)]

    # Iterate over each image in the batch
    for img in modified_batch:
        for x, y in square_coords:
            # Apply the mask
            img[:, x:x+s, y:y+s] = 0

            # Update the binary mask
            binary_mask[:, :, x:x+s, y:y+s] = 1

    return modified_batch, binary_mask
