from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from scipy.stats import randint

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    labels_df = pd.read_csv(config["data_dir"] / "train_labels.csv", dtype={"ID": str})

    # Stratified split
    distance_bins = pd.qcut(distances, q=5, labels=False, duplicates="drop")
    idx = np.arange(len(images))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=distance_bins
    )
    X_train, X_val = images[idx_train], images[idx_val]
    y_train, y_val = distances[idx_train], distances[idx_val]

    # Pipeline with explicit names
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA()),
        ("model",  KNeighborsRegressor(
            n_neighbors=2,
            weights="distance",
            p=2,
        )),
    ])

    param_dist = {
        "pca__n_components":      randint(40, 80),
        "model__n_neighbors":     randint(2, 5),
        "model__weights":         ["distance"],
        "model__p":               [1, 2],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=40,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        random_state=42,
        refit=True,
    )

    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_*100:.2f} cm")

    y_pred = search.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae*100:.2f} cm")

    # --- Error analysis ---
    errors = np.abs(y_pred - y_val)
    error_bins = [0, 1.2, 1.5, 1.8, 2.2, 10]
    names      = ["very close", "close", "middle", "far", "very far"]
    for low, high, name in zip(error_bins[:-1], error_bins[1:], names):
        mask = (y_val >= low) & (y_val < high)
        if mask.sum() > 0:
            print(f"{name}: count={mask.sum()}, MAE={errors[mask].mean()*100:.2f} cm")

    # # --- Refit on ALL data, predict test set ---
    # search.best_estimator_.fit(images, distances)
    # test_images = load_test_dataset(config)
    # test_pred = search.best_estimator_.predict(test_images)
    # save_results(test_pred)
    # print("[INFO]: Predictions saved to prediction.csv")