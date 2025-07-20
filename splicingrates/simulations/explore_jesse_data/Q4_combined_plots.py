import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def combine_plots_vertical(folder_path):
    """Combine all PNG files in a folder into a vertical stack."""

    # Get list of PNG files
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    n_images = len(png_files)

    # Create figure with one column and n_images rows
    fig, axes = plt.subplots(n_images, 1, figsize=(8, 6 * n_images))

    # Convert to array if only one image
    if n_images == 1:
        axes = np.array([axes])

    # Plot each image
    for idx, fname in enumerate(png_files):
        img = Image.open(os.path.join(folder_path, fname))
        axes[idx].imshow(np.array(img))
        axes[idx].set_title(fname)
        axes[idx].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(folder_path, 'combined_plots.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Combined plot saved to: {output_path}")

if __name__ == '__main__':
    folder_path = 'path/to/folder'
    combine_plots_vertical(folder_path)