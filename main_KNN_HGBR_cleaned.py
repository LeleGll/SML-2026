from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.stats import uniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer

def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))

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

    # KNN Pipeline
    KNN_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=47)),
        ("model",  KNeighborsRegressor(
            n_neighbors=2,
            weights="distance",
            p=2,
        )),
    ])

    # Hist Gradient Boost Regressor
    HGBR_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=47)),
        ("model",  HistGradientBoostingRegressor(
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

    KNN_pipeline.fit(X_train, y_train)

    # Close Range for Ridge
    close_mask_train = y_train < 1.2
    X_train_close = X_train[close_mask_train]
    y_train_close = y_train[close_mask_train]
    print(f"[INFO]: Tuning Ridge on {len(y_train_close)} very close samples.")

    # Param Grid
    param_dist = {
        "pca__n_components": [30, 40, 50, 60],
        "model__n_neighbors": [2, 3, 5, 7],
        "model__weights": ["distance", "uniform"],
        "model__p": [1, 2]
    }

    median_scorer = make_scorer(
        median_absolute_error,
        greater_is_better=False
    )

    # Random search for Ridge
    randomized_search = RandomizedSearchCV(
        KNN_pipeline,
        param_distributions=param_dist,
        cv=5,
        scoring=median_scorer,
        n_jobs=-1,
        random_state=42
    )

    # Only close range fits
    close_mask_train = y_train < 1.2

    randomized_search.fit(X_train[close_mask_train], y_train[close_mask_train])

    print("Best params:", randomized_search.best_params_)
    print("Best CV MAE:", -randomized_search.best_score_)

    # KNN predict all
    y_pred_KNN = KNN_pipeline.predict(X_val)

    # HGBR close Range
    HGBR_close = randomized_search.best_estimator_
    HGBR_pred = HGBR_close.predict(X_val)
    
    # Replace close predictions
    close_mask_val = y_pred_KNN < 1.2

    y_pred_hybrid = y_pred_KNN.copy()
    y_pred_hybrid[close_mask_val] = HGBR_pred[close_mask_val]

    mae_KNN = mean_absolute_error(y_val, y_pred_KNN)
    mae_hybrid = mean_absolute_error(y_val, y_pred_hybrid)
    print(f"KNN MAE: {mae_KNN*100:.2f} cm")
    print(f"Hybrid MAE: {mae_hybrid*100:.2f} cm")
    print(f"KNN + HGBR close MAE: {mae_hybrid*100:.2f} cm")

    # print(f"Best Ridge params: {ridge_search.best_params_}")
    # print(f"Best Ridge CV MAE: {-ridge_search.best_score_*100:.2f} cm")


    # # Override very close with Ridge
    # close_mask_val = y_pred < 1.2
    # y_pred[close_mask_val] = ridge_pipeline.predict(X_val[close_mask_val])

    # mae = mean_absolute_error(y_val, y_pred)
    # print(f"Validation MAE: {mae*100:.2f} cm")

    # y_pred = KNN_pipeline.predict(X_val)
    # mae = mean_absolute_error(y_val, y_pred)
    # print(f"Validation MAE: {mae*100:.2f} cm")

    # --- Error analysis ---
    errors = np.abs(y_pred_hybrid - y_val)
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