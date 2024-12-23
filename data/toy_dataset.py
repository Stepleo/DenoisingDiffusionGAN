import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GaussianGridDataset(Dataset):
    """
    Dataset for 25-Gaussians, represented as points on a grid.
    """

    def __init__(self, grid_size: int = 28, num_samples: int = 100000, std_dev: float = 0.05):
        """
        Args:
            grid_size: Size of the square grid for the images.
            num_samples: Number of data points to sample from the dataset.
            std_dev: Standard deviation of each Gaussian.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.std_dev = std_dev

        # 25 Gaussian centers in a grid, scaled to fit within the grid
        self.centers = np.array([(x, y) for x in range(-2, 3) for y in range(-2, 3)], dtype=np.float32)
        self.centers *= (grid_size / 6)  # Scale centers to fit within grid, leaving margins

        # Generate samples
        self.data = []
        for _ in range(num_samples):
            center = self.centers[np.random.choice(len(self.centers))]
            point = np.random.normal(loc=center, scale=std_dev, size=center.shape)
            self.data.append(point)

        self.data = np.array(self.data, dtype=np.float32)

    def _point_to_grid(self, point):
        """
        Convert a continuous point to a discrete grid representation.

        Args:
            point: A 2D point (x, y).

        Returns:
            A 2D binary grid with a single active pixel.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        x, y = point
        # Convert continuous point to nearest integer grid location
        grid_x = np.clip(int(round(x)), 0, self.grid_size - 1)
        grid_y = np.clip(int(round(y)), 0, self.grid_size - 1)
        grid[grid_x, grid_y] = 1.0
        return grid

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        point = self.data[idx]
        grid = self._point_to_grid(point)
        return torch.tensor(grid, dtype=torch.float32)


def get_25gaussians_grid_dataloader(batch_size: int = 256, grid_size: int = 28, num_samples: int = 100000, std_dev: float = 0.05):
    """
    Creates a DataLoader for the 25-Gaussians dataset as grid-based images.

    Args:
        batch_size: Batch size for the DataLoader.
        grid_size: Size of the square grid for the images.
        num_samples: Number of data points in the dataset.
        std_dev: Standard deviation for each Gaussian.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = GaussianGridDataset(grid_size=grid_size, num_samples=num_samples, std_dev=std_dev)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

