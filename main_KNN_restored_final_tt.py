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

    # Stratified split
    distance_bins = pd.qcut(distances, q=5, labels=False, duplicates="drop")
    idx = np.arange(len(images))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=distance_bins
    )
    X_train, X_val = images[idx_train], images[idx_val]
    y_train, y_val = distances[idx_train], distances[idx_val]

    # Pipeline with best params
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=59)),
        ("model",  KNeighborsRegressor(
            n_neighbors=2,
            weights="distance",
            p=2,
        )),
    ])

    # Fit on train, evaluate on val
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae*100:.2f} cm")

    # Refit on ALL data, predict Kaggle test set
    pipeline.fit(images, distances)
    test_images = np.array(load_test_dataset(config))
    test_pred = pipeline.predict(test_images)
    save_results(test_pred)
    print("[INFO]: Predictions saved to prediction.csv")