from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = "data/train_labels.csv"

# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    # print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    # print(f"Object Type: {type(images)}")
    # print(f"Current Shape: {images.shape}")
    # print(f"Data Type: {images.dtype}")

    # TODO: Your implementation starts here

    # 1. Prepare Data
    red_mean = images[:, 0::3].mean(axis=1)
    green_mean = images[:, 1::3].mean(axis=1)
    blue_mean = images[:, 2::3].mean(axis=1)
    avg_intensity = images.mean(axis=1)

    # 2. Create a 3x2 Grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # --- ROW 1: Distribution & Overall Average ---
    # Top Left: The Histogram you just asked for
    axes[0, 0].hist(distances, bins=30, edgecolor='black', density=True, alpha=0.7, label='Data Density')
    sns.kdeplot(distances, color='blue', linewidth=2, label='Density Curve', ax=axes[0, 0])
    axes[0, 0].axvline(np.mean(distances), color='red', linestyle='dashed', label='Mean')
    axes[0, 0].set_title('Distribution of Distance Labels')
    axes[0, 0].legend()

    # Top Right: Overall Intensity vs Distance
    axes[0, 1].scatter(avg_intensity, distances, color='gray', alpha=0.4)
    # Add trend line
    m, b = np.polyfit(avg_intensity, distances, 1)
    axes[0, 1].plot(avg_intensity, m*avg_intensity + b, color='black', linewidth=2, label='Trend')
    axes[0, 1].set_title('Overall Avg Intensity vs Distance')

    # --- ROW 2: Individual Colors ---
    # Middle Left: Red
    axes[1, 0].scatter(red_mean, distances, color='red', alpha=0.4)
    m_r, b_r = np.polyfit(red_mean, distances, 1)
    axes[1, 0].plot(red_mean, m_r*red_mean + b_r, color='darkred', linewidth=2)
    axes[1, 0].set_title('Red Mean vs Distance')

    # Middle Right: Green
    axes[1, 1].scatter(green_mean, distances, color='green', alpha=0.4)
    m_g, b_g = np.polyfit(green_mean, distances, 1)
    axes[1, 1].plot(green_mean, m_g*green_mean + b_g, color='darkgreen', linewidth=2)
    axes[1, 1].set_title('Green Mean vs Distance')

    # --- ROW 3: Blue ---
    # Bottom Left: Blue
    axes[2, 0].scatter(blue_mean, distances, color='blue', alpha=0.4)
    m_b, b_b = np.polyfit(blue_mean, distances, 1)
    axes[2, 0].plot(blue_mean, m_b*blue_mean + b_b, color='darkblue', linewidth=2)
    axes[2, 0].set_title('Blue Mean vs Distance')

    # # Top Right: Overall Intensity vs Distance
    # axes[0, 1].scatter(avg_intensity, distances, color='gray', alpha=0.4)
    # axes[0, 1].set_title('Overall Avg Intensity vs Distance')

    # # --- ROW 2: Individual Colors ---
    # # Middle Left: Red
    # axes[1, 0].scatter(red_mean, distances, color='red', alpha=0.4)
    # axes[1, 0].set_title('Red Mean vs Distance')

    # # Middle Right: Green
    # axes[1, 1].scatter(green_mean, distances, color='green', alpha=0.4)
    # axes[1, 1].set_title('Green Mean vs Distance')

    # # --- ROW 3: Blue & (Empty or Extra) ---
    # # Bottom Left: Blue
    # axes[2, 0].scatter(blue_mean, distances, color='blue', alpha=0.4)
    # axes[2, 0].set_title('Blue Mean vs Distance')

    # Bottom Right: Leave empty or use for Log-Transformed version
    axes[2, 1].axis('off') 

    # 3. Global Labels & Formatting
    for i in range(3):
        for j in range(2):
            if not (i == 0 and j == 0): # Don't apply scatter labels to the histogram
                axes[i, j].set_xlabel('Intensity Value')
                axes[i, j].set_ylabel('Distance (m)')

    plt.tight_layout()
    plt.show()