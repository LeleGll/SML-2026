from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

if __name__ == "__main__":
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # --- 1. Train / validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )

    # --- 2. Pipeline: Scale → PCA → KNN ---
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=100, random_state=42)),
        ("model",  KNeighborsRegressor()),
    ])

    # --- 3. Hyperparameter grid ---
    param_grid = {
        "pca__n_components": [50, 100, 150, 200],
        "model__n_neighbors": [3, 5, 7, 10, 15],
        "model__weights":     ["uniform", "distance"],
        "model__metric":      ["euclidean", "manhattan"],
    }

    # --- 4. Grid search with 5-fold CV ---
    search = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"\n[INFO]: Best params: {search.best_params_}")
    print(f"[INFO]: Best CV MAE: {-search.best_score_*100:.2f} cm")

    # --- 5. Evaluate on validation set ---
    val_pred = search.predict(X_val)
    print_results(y_val, val_pred)

    # --- 6. Retrain best config on ALL data, predict test set ---
    best_pipe = search.best_estimator_
    best_pipe.fit(images, distances)

    test_images = load_test_dataset(config)
    test_pred = best_pipe.predict(test_images)
    save_results(test_pred)
    print("[INFO]: Predictions saved to prediction.csv")