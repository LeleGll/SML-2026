import sys
print(sys.executable)
from utils import load_config, load_dataset, load_test_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor


file_path = "C:/Users/guido/SML Projekte 2026/Projekt 1/data/train_labels.csv"
# sklearn imports...
# SVRs are not allowed in this project.

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # TODO: Your implementation starts here
    # flatten images
    images = images.reshape(len(images), -1)

    # creating bins
    distance_bins = np.digitize(distances, bins=[1.2, 1.5, 1.7, 1.9, 2.2])

    # normal Train Test Split on 20% of Data
    X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
        images, 
        distances,
        distance_bins, 
        test_size=0.2, 
        random_state=0,
        stratify=distance_bins
    )

    # Creating threshold for ensemble
    close_threshold = 1.2

    close_mask = y_train < close_threshold
    mid_far_mask = y_train >= close_threshold
    
    X_train_close = X_train[close_mask]
    y_train_close = y_train[close_mask]

    X_train_mid_far = X_train[mid_far_mask]
    y_train_mid_far = y_train[mid_far_mask]

    # # Log transform distance labels
    # y_train_log = np.log1p(y_train)
    # y_test_log = np.log1p(y_test)

    # # Creating bins for stratisfied K-Folds
    # y_bins = np.digitize(y_train, bins=np.linspace(y_train.min(), y_train.max(), 10))

    # Bins for mid and far
    bins_train_mid_far = bins_train[mid_far_mask]

    # Set up Stratified KFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = skf.split(X_train_mid_far, bins_train_mid_far)

    # Initializing KNN Pipeline
    KNN_pipeline = make_pipeline(
                        preprocessing.RobustScaler(), 
                        PCA(n_components=150), 
                        KNeighborsRegressor(n_neighbors=5, weights='distance'))
    
    # Initializing GB Pipeline
    GB_pipeline = make_pipeline(
        preprocessing.RobustScaler(),
        PCA(n_components=80),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=200,
            learning_rate=0.05,
            early_stopping=True,
            max_leaf_nodes=31,
            random_state=0
        )
    )

    # Initializing Gate Model
    gate_model = make_pipeline(
        preprocessing.RobustScaler(),
        PCA(n_components=80),
        KNeighborsRegressor(n_neighbors=5, weights="distance")
    )
    
    # # Log transformer
    # transformer = TransformedTargetRegressor(
    #     regressor=KNN_pipeline,
    #     func=np.log1p,
    #     inverse_func=np.expm1
    # )
    
    # Parameters for KNN_Search
    KNN_param_distributions = {
        'pca__n_components': [70, 75, 80, 85],
        'kneighborsregressor__n_neighbors' : [2, 3, 4, 5, 6],
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p':[1,2]
    }

    # Parameters for GB_Search
    GB_param_distributions = {
        # PCA
        "pca__n_components": [50, 75, 100],

        # Learning behavior
        "histgradientboostingregressor__learning_rate": [0.03, 0.05, 0.1],

        # Number of trees
        "histgradientboostingregressor__max_iter": [100, 200],

        # Tree complexity
        "histgradientboostingregressor__max_leaf_nodes": [15, 31, 63],

        # Regularization
        "histgradientboostingregressor__min_samples_leaf": [10, 20, 50],
        "histgradientboostingregressor__l2_regularization": [0.0, 0.01, 0.1]
    }

    # KNN_search
    KNN_random_search = RandomizedSearchCV(
        KNN_pipeline, 
        KNN_param_distributions,
        n_iter=80, 
        cv=cv_splits,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    # GB_search
    GB_random_search = RandomizedSearchCV(
        GB_pipeline,
        GB_param_distributions,
        n_iter=40,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=0
    )

    GB_random_search.fit(X_train_close, y_train_close)
    KNN_random_search.fit(X_train_mid_far, y_train_mid_far)
    gate_model.fit(X_train, y_train)

    # Use the best model
    KNN_best_model = KNN_random_search.best_estimator_
    GB_best_model = GB_random_search.best_estimator_

    # Set up Global Gate
    global_model = gate_model
    rough_pred = global_model.predict(X_test)

    KNN_pred = KNN_best_model.predict(X_test)
    GB_pred = GB_best_model.predict(X_test)

    final_pred = KNN_pred.copy()

    close_region = rough_pred < 1.2
    border_region = (rough_pred >= 1.2) & (rough_pred < 1.5)

    final_pred[close_region] = 0.3 * KNN_pred[close_region] + 0.7 * GB_pred[close_region]
    final_pred[border_region] = 0.7 * KNN_pred[border_region] + 0.3 * GB_pred[border_region]

    # Calculate MAE
    mae = mean_absolute_error(y_test, final_pred)
    print(f"Ensemble MAE: {mae:.4f} meters")

    # Error calculation of distances
    errors = np.abs(final_pred - y_test)
    bins = [0, 1.2, 1.6, 2.0, 10]
    names = ["very close", "close-mid", "around 1.8", "far"]
    for low, high, name, in zip (bins[:-1], bins[1:], names):
        mask = (y_test >= low) & (y_test < high)
        print(name, "count:", mask.sum(), "MAE:", errors[mask].mean())

    # Show the first 5 results
    print("\n--- Final Distance Outputs ---")
    for i in range(5):
        diff = abs(final_pred[i] - y_test[i])
        print(f"Image {i}:")
        print(f"    Predicted = {final_pred[i]:.4f} m")
        print(f"    Actual    = {y_test[i]:.4f} m")
        print(f"    Error     = {diff:.4f} m")
        print()
    print(f"Mean Absolute Error: {mae:.4f} meters")

    # --- THE SUBMISSION CODE GOES HERE ---
    print("[INFO]: Generating Kaggle submission...")
    test_images_raw = load_test_dataset(config)
    test_images = np.array(test_images_raw)
    test_images_flat = test_images.reshape(len(test_images), -1)

    # predict on the test set
    y_kaggle_final = global_model.predict(test_images_flat)

    # ... rest of the code I gave you ...
    submission_df = pd.DataFrame({
    "ID": [f"{i:03d}" for i in range(len(y_kaggle_final))],
    "Distance": y_kaggle_final
    })

    submission_df.to_csv("KNN_Pipeline_GridSearch_submission.csv", index=False)
    
    print(f"[SUCCESS]: Created submission with {len(submission_df)} rows!")

    