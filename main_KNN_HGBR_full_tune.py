from utils import load_config, load_dataset
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def median_absolute_error_custom(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))


if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")


    h = 300 // config["downsample_factor"]
    w = 300 // config["downsample_factor"]

    processed_images = []

    for img in images:
        img_2d = img.reshape(h, w)
        img_2d = img_2d / 255.0

        img_small = img_2d[::3, ::3]

        row_means = img_2d.mean(axis=1)
        col_means = img_2d.mean(axis=0)

        combined = np.concatenate([
            img_small.flatten(),
            row_means,
            col_means
        ])

        processed_images.append(combined)

    images = np.array(processed_images)

    # Stratified split
    distance_bins = pd.qcut(distances, q=5, labels=False, duplicates="drop")
    idx = np.arange(len(images))

    idx_train, idx_val = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=distance_bins
    )

    X_train, X_val = images[idx_train], images[idx_val]
    y_train, y_val = distances[idx_train], distances[idx_val]

    # -----------------------------
    # 1) Fixed HGBR close model
    # -----------------------------
    HGBR_close_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=47)),
        ("model", HistGradientBoostingRegressor(
            loss="absolute_error",
            random_state=42,
            early_stopping=True,
            min_samples_leaf=10,
            max_leaf_nodes=63,
            max_iter=300,
            learning_rate=0.03,
            l2_regularization=0.0
        )),
    ])

    close_mask_train = y_train < 1.2

    HGBR_close_pipeline.fit(
        X_train[close_mask_train],
        y_train[close_mask_train]
    )

    bin_edges = [1.2, 1.5, 1.8, 2.2]

    y_train_bins = np.digitize(y_train, bins=bin_edges)
    y_val_bins = np.digitize(y_val, bins=bin_edges)

    bin_classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=47)),
        ("model", HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.0
        ))
    ])

    bin_classifier.fit(X_train, y_train_bins)

    # -----------------------------
    # 2) Tune global KNN on ALL train samples
    #    using median absolute error
    # -----------------------------
    KNN_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=40)),
        ("model", KNeighborsRegressor(
            weights='distance',
            p=2,
            n_neighbors=2
        ))
    ])

    knn_param_dist = {
        "pca__n_components": [30, 40, 47, 50, 60],
        "model__n_neighbors": [2, 3, 4, 5, 7],
        "model__weights": ["distance", "uniform"],
        "model__p": [1, 2]
    }

    knn_search = RandomizedSearchCV(
        KNN_pipeline,
        param_distributions=knn_param_dist,
        n_iter=40,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )

    knn_search.fit(X_train, y_train)

    print("Best KNN params:", knn_search.best_params_)
    print("Best KNN CV MAE:", -knn_search.best_score_)

    best_knn = knn_search.best_estimator_

    # -----------------------------
    # 3) Predict validation
    # -----------------------------

    y_pred_knn = best_knn.predict(X_val)
    y_pred_hgbr_close = HGBR_close_pipeline.predict(X_val)

    # Hybrid: replace close predictions only
    pred_bins = bin_classifier.predict(X_val)
    use_hgbr = pred_bins == 0

    y_pred_hybrid = y_pred_knn.copy()
    y_pred_hybrid[use_hgbr] = y_pred_hgbr_close[use_hgbr]

    # -----------------------------
    # 4) Evaluate
    # -----------------------------

    print("Bin classifier accuracy:", accuracy_score(y_val_bins, pred_bins))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_bins, pred_bins))

    mae_knn = mean_absolute_error(y_val, y_pred_knn)
    mae_hybrid = mean_absolute_error(y_val, y_pred_hybrid)

    print(f"KNN MAE: {mae_knn*100:.2f} cm")
    print(f"Hybrid MAE: {mae_hybrid*100:.2f} cm")

    print("\n--- Overall MAE ---")
    print(f"KNN:    {mean_absolute_error(y_val, y_pred_knn)*100:.2f} cm")
    print(f"HGBR:   {mean_absolute_error(y_val, y_pred_hgbr_close)*100:.2f} cm")
    print(f"Hybrid: {mean_absolute_error(y_val, y_pred_hybrid)*100:.2f} cm")

    true_close = y_val < 1.2

    y_pred_oracle = y_pred_knn.copy()
    y_pred_oracle[true_close] = y_pred_hgbr_close[true_close]

    print("Oracle MAE:", mean_absolute_error(y_val, y_pred_oracle) * 100)


    # --------------------------------------------
    # 5) Error analysis by distance bin
    # --------------------------------------------
    predictions = {
        "KNN": y_pred_knn,
        "HGBR": y_pred_hgbr_close,
        "Hybrid": y_pred_hybrid,
    }

    error_bins = [0, 1.2, 1.5, 1.8, 2.2, 10]
    names = ["very close", "close", "middle", "far", "very far"]

    print("\n--- Error analysis by distance bin ---")
    print(f"{'Bin':<12} {'Count':>6} {'KNN':>10} {'HGBR':>10} {'Hybrid':>10} ")
    print("-" * 64)

    for low, high, name in zip(error_bins[:-1], error_bins[1:], names):
        mask = (y_val >= low) & (y_val < high)

        if mask.sum() > 0:
            knn_mae = np.mean(np.abs(y_pred_knn[mask] - y_val[mask])) * 100
            hgbr_mae = np.mean(np.abs(y_pred_hgbr_close[mask] - y_val[mask])) * 100
            hybrid_mae = np.mean(np.abs(y_pred_hybrid[mask] - y_val[mask])) * 100

            print(
                f"{name:<12} {mask.sum():>6} "
                f"{knn_mae:>10.2f} "
                f"{hgbr_mae:>10.2f} "
                f"{hybrid_mae:>10.2f} "
            )
