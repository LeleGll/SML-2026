from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import blur_effect
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    labels = pd.read_csv(config["data_dir"] / "train_labels.csv", dtype={"ID": str})

    distance_bins = pd.qcut(distances, q=5, labels=False, duplicates="drop")
    idx = np.arange(len(images))
    idx_train, idx_val = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=distance_bins
    )

    X_train, X_val = images[idx_train], images[idx_val]
    y_train, y_val = distances[idx_train], distances[idx_val]
    ids_val = labels["ID"].values[idx_val]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=150)),
        ("model",  GradientBoostingRegressor(random_state=42)),
    ])

    param_dist = {
        "pca__n_components":       randint(50, 200),
        "model__n_estimators":     randint(100, 500),
        "model__learning_rate":    uniform(0.01, 0.14),
        "model__max_depth":        randint(3, 6),
        "model__subsample":        uniform(0.7, 0.3),
        "model__min_samples_leaf": randint(3, 10),
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=60,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        random_state=42,
        refit=True,
    )
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train < 1.2] = 4.0
    search.fit(X_train, y_train, model__sample_weight=sample_weights)
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

    # --- Very close image visualization ---
    very_close_idx = np.where(distances < 1.2)[0]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, i in zip(axes.flat, very_close_idx[:10]):
        img_id = labels["ID"].values[i]
        img = Image.open(config["data_dir"] / "train_images" / f"{img_id}.png")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{distances[i]:.2f}m")
        ax.axis("off")
    plt.suptitle("Very close obstacles (<1.2m)")
    plt.show()

    # --- Blur analysis ---
    blurriness = []
    for img_id in ids_val:
        img = np.array(Image.open(
            config["data_dir"] / "train_images" / f"{img_id}.png"
        ).convert("L"))
        blurriness.append(blur_effect(img))

    plt.scatter(blurriness, errors, alpha=0.3)
    plt.xlabel("Blur score")
    plt.ylabel("Prediction error (m)")
    plt.title("Does blur correlate with error?")
    plt.show()

    # --- Refit on ALL data, predict test set ---
    search.best_estimator_.fit(images, distances)
    test_images = load_test_dataset(config)
    test_pred = search.best_estimator_.predict(test_images)
    save_results(test_pred)
    print("[INFO]: Predictions saved to prediction.csv")