from utils import load_config, load_dataset, load_test_dataset, save_results
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # --- 3-way stratified split ---
    distance_bins = pd.qcut(distances, q=5, labels=False, duplicates="drop")
    idx = np.arange(len(images))

    # First split off 20% as held-out test
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=distance_bins
    )

    # Split remaining 80% into train/val
    distance_bins_trainval = pd.qcut(
        distances[idx_trainval], q=5, labels=False, duplicates="drop"
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.2, random_state=42,
        stratify=distance_bins_trainval
    )

    X_train = images[idx_train]
    X_val   = images[idx_val]
    X_test  = images[idx_test]
    y_train = distances[idx_train]
    y_val   = distances[idx_val]
    y_test  = distances[idx_test]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # --- Pipeline ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=59)),
        ("model",  KNeighborsRegressor(
            n_neighbors=2,
            weights="distance",
            p=2,
        )),
    ])

    # --- Fit on train, evaluate on val ---
    pipeline.fit(X_train, y_train)
    y_pred_val = pipeline.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    print(f"Validation MAE: {mae_val*100:.2f} cm")

    # --- Final unbiased evaluation on held-out test set ---
    y_pred_test = pipeline.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print(f"Test MAE (unbiased): {mae_test*100:.2f} cm")

    # --- Stability check across seeds ---
    print("\n--- Stability check ---")
    for seed in [42, 0, 1, 7, 123]:
        idx_tr, idx_v = train_test_split(
            idx_trainval, test_size=0.2, random_state=seed,
            stratify=distance_bins_trainval
        )
        pipeline.fit(images[idx_tr], distances[idx_tr])
        pred = pipeline.predict(images[idx_v])
        mae = mean_absolute_error(distances[idx_v], pred)
        print(f"Seed {seed}: MAE = {mae*100:.2f} cm")

    # --- Refit on ALL data, predict Kaggle test set ---
    pipeline.fit(images, distances)
    test_images = np.array(load_test_dataset(config))
    test_pred = pipeline.predict(test_images)
    save_results(test_pred)
    print("\n[INFO]: Predictions saved to prediction.csv")