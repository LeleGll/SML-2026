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
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from skimage.filters import sobel
from PIL import Image
import matplotlib.pyplot as plt

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
    h = 300 // config["downsample_factor"]
    w = 300 // config["downsample_factor"]

    processed_images = []

    for img in images:
        img_2d = img.reshape(h, w)

        edges = sobel(img_2d)

        bottom = img_2d[2*h//3:, :]
        left_edge = img_2d[:, :w//6]
        right_edge = img_2d[:, 5*w//6:]

        extra_features = np.array([
            bottom.mean(), bottom.std(),
            left_edge.mean(), left_edge.std(),
            right_edge.mean(), right_edge.std()
        ])

        combined = np.concatenate([
            img_2d.flatten(),
            edges.flatten(),
            extra_features
        ])

        processed_images.append(combined)

    images = np.array(processed_images)

    # creating bins
    distance_bins = np.digitize(distances, bins=[1.2, 1.5, 1.7, 1.9, 2.2])

    # creating ids to find problem picture
    labels_df = pd.read_csv(config["data_dir"] / "train_labels.csv", dtype={"ID": str})
    ids = labels_df["ID"].values

    # normal Train Test Split on 20% of Data
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        images, 
        distances,
        ids, 
        test_size=0.2, 
        random_state=0,
        stratify=distance_bins
    )



    # Initializing KNN Pipeline
    KNN_pipeline = make_pipeline(
                        preprocessing.RobustScaler(), 
                        PCA(n_components=100), 
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

    # Initializing Ridge Pipeline
    ridge_pipeline = make_pipeline(
        preprocessing.StandardScaler(),
        PCA(n_components=100),
        Ridge(alpha=10.0)
    )

    # Initializing RandomForest Pipeline
    rf_pipeline = make_pipeline(
        preprocessing.RobustScaler(),
        PCA(n_components=100),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=0,
            n_jobs=2
            )
    )

    # Ensemble
    ensemble = VotingRegressor(
        estimators=[
            ("knn", KNN_pipeline),
            ("ridge", ridge_pipeline),
            ("gb", GB_pipeline),
            ("rf", rf_pipeline)
        ],
        weights=[2, 1, 3, 2],
        n_jobs=1
    )

    ensemble.fit(X_train, y_train)

    # Outlier detection
    train_pred = ensemble.predict(X_train)
    train_errors = np.abs(train_pred - y_train)

    threshold = np.percentile(train_errors, 95)
    mask = train_errors < threshold

    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    # Retrain
    ensemble.fit(X_train_clean, y_train_clean)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ensemble MAE: {mae:.4f} meters")

    # 🔍 Error analysis by distance bins
    errors = np.abs(y_pred - y_test)

    bins = [0, 1.2, 1.5, 1.8, 2.2, 10]
    names = ["very close", "close", "middle", "far", "very far"]

    for low, high, name in zip(bins[:-1], bins[1:], names):
        mask = (y_test >= low) & (y_test < high)
    
        if mask.sum() > 0:  # avoid empty bins
            print(f"{name}: count={mask.sum()}, MAE={errors[mask].mean():.4f}")

    worst_idx = np.argsort(errors)[-10:]

    print("\nWorst predictions:")
    for i in worst_idx:
        print(f"Idx {i}: Pred={y_pred[i]:.3f}, True={y_test[i]:.3f}, Error={errors[i]:.3f}")

    for i in worst_idx:
        print(f"ID {id_test[i]}: Pred={y_pred[i]:.3f}, True={y_test[i]:.3f}")

    # Plotting errors
    img_id = id_test[i]
    img_path = config["data_dir"] / "train_images" / f"{img_id}.png"

    for i in worst_idx:
        img_id = id_test[i]
        img_path = config["data_dir"] / "train_images" / f"{img_id}.png"

        img = Image.open(img_path)

        plt.imshow(img, cmap="gray")
        plt.title(f"ID={img_id}, Pred={y_pred[i]:.2f}, True={y_test[i]:.2f}")
        plt.axis("off")
        plt.show()

    # # Initializing Gate Model
    # gate_model = make_pipeline(
    #     preprocessing.RobustScaler(),
    #     PCA(n_components=80),
    #     KNeighborsRegressor(n_neighbors=5, weights="distance")
    # )
    
    # # Log transformer
    # transformer = TransformedTargetRegressor(
    #     regressor=KNN_pipeline,
    #     func=np.log1p,
    #     inverse_func=np.expm1
    # )


    # GB_random_search.fit(X_train_close, y_train_close)
    # KNN_random_search.fit(X_train_mid_far, y_train_mid_far)
    # gate_model.fit(X_train, y_train)

    # # Use the best model
    # KNN_best_model = KNN_random_search.best_estimator_
    # GB_best_model = GB_random_search.best_estimator_

    # # Set up Global Gate
    # global_model = gate_model
    # rough_pred = global_model.predict(X_test)

    # KNN_pred = KNN_best_model.predict(X_test)
    # GB_pred = GB_best_model.predict(X_test)

    # final_pred = KNN_pred.copy()

    # close_region = rough_pred < 1.2
    # border_region = (rough_pred >= 1.2) & (rough_pred < 1.5)

    # final_pred[close_region] = 0.3 * KNN_pred[close_region] + 0.7 * GB_pred[close_region]
    # final_pred[border_region] = 0.7 * KNN_pred[border_region] + 0.3 * GB_pred[border_region]

    # # Error calculation of distances
    # errors = np.abs(final_pred - y_test)
    # bins = [0, 1.2, 1.6, 2.0, 10]
    # names = ["very close", "close-mid", "around 1.8", "far"]
    # for low, high, name, in zip (bins[:-1], bins[1:], names):
    #     mask = (y_test >= low) & (y_test < high)
    #     print(name, "count:", mask.sum(), "MAE:", errors[mask].mean())

    # # Show the first 5 results
    # print("\n--- Final Distance Outputs ---")
    # for i in range(5):
    #     diff = abs(final_pred[i] - y_test[i])
    #     print(f"Image {i}:")
    #     print(f"    Predicted = {final_pred[i]:.4f} m")
    #     print(f"    Actual    = {y_test[i]:.4f} m")
    #     print(f"    Error     = {diff:.4f} m")
    #     print()
    # print(f"Mean Absolute Error: {mae:.4f} meters")

    # # --- THE SUBMISSION CODE GOES HERE ---
    # print("[INFO]: Generating Kaggle submission...")
    # test_images_raw = load_test_dataset(config)
    # test_images = np.array(test_images_raw)
    # test_images_flat = test_images.reshape(len(test_images), -1)

    # # predict on the test set
    # y_kaggle_final = global_model.predict(test_images_flat)

    # # ... rest of the code I gave you ...
    # submission_df = pd.DataFrame({
    # "ID": [f"{i:03d}" for i in range(len(y_kaggle_final))],
    # "Distance": y_kaggle_final
    # })

    # submission_df.to_csv("KNN_Pipeline_GridSearch_submission.csv", index=False)
    
    # print(f"[SUCCESS]: Created submission with {len(submission_df)} rows!")

# # Creating threshold for ensemble
    # close_threshold = 1.2

    # close_mask = y_train < close_threshold
    # mid_far_mask = y_train >= close_threshold
    
    # X_train_close = X_train[close_mask]
    # y_train_close = y_train[close_mask]

    # X_train_mid_far = X_train[mid_far_mask]
    # y_train_mid_far = y_train[mid_far_mask]

    # # # Log transform distance labels
    # # y_train_log = np.log1p(y_train)
    # # y_test_log = np.log1p(y_test)

    # # Creating bins for stratisfied K-Folds
    # y_bins = np.digitize(y_train, bins=np.linspace(y_train.min(), y_train.max(), 10))

    # # # Bins for mid and far
    # # bins_train_mid_far = bins_train[mid_far_mask]

    # # Set up Stratified KFold
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_splits = list(skf.split(X_train, bins_train))