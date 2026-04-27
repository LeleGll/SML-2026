import sys
print(sys.executable)
from utils import load_config, load_dataset, load_test_dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler


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
        img_2d = img_2d / 255.0

        img_small = img_2d[::2, ::2]
        combined = img_small.flatten()

        processed_images.append(combined)

    images = np.array(processed_images)

    # creating bins
    distance_bins = np.digitize(distances, bins=[1.2, 1.5, 1.7, 1.9, 2.2])

    # creating ids to find problem picture
    labels_df = pd.read_csv(config["data_dir"] / "train_labels.csv", dtype={"ID": str})
    ids = labels_df["ID"].values

    # Initializing Pipeline KNN
    KNN_pipeline = make_pipeline(
                        preprocessing.StandardScaler(), 
                        PCA(n_components=40),
                        KNeighborsRegressor(n_neighbors=2, weights='distance', p=2))
    
    Ridge_pipeline = make_pipeline(
                        preprocessing.StandardScaler(), 
                        PCA(n_components=200), 
                        Ridge(alpha=1438.44988828766)
                    )
    
    rf_pipeline = make_pipeline(
                        RobustScaler(),
                        PCA(n_components=50),
                        RandomForestRegressor(
                            max_depth=None, 
                            max_features='sqrt',
                            min_samples_leaf=1,
                            min_samples_split=2,
                            n_estimators=200
                        )
                        )
    
    # set up grid search
    # param_distributions = {
    #     'pca__n_components': [40],
    #     'kneighborsregressor__n_neighbors' : [2],
    #     'kneighborsregressor__weights': ["distance"],
    #     'kneighborsregressor__p':[2]
    # }

    # Set up Seeds
    for seed in [0, 1, 2, 3, 4, 42]:
        print(f"\n===== SEED {seed} =====")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            images,
            distances,
            test_size=0.2,
            random_state=seed,
            stratify=distance_bins
        )
    
        y_train_log = np.log1p(y_train)

        # # Fit and Predict
        # random_search = GridSearchCV(
        #     KNN_pipeline, 
        #     param_distributions,
        #     cv=5,
        #     scoring='neg_mean_absolute_error',
        #     n_jobs=-1  
        # )

        # random_search.fit(X_train, y_train_log)
        # print("Best best p:", random_search.best_params_["kneighborsregressor__p"])
        # print("Best CV MAE:", -random_search.best_score_)

        # # Use the best model
        # best_model = random_search.best_estimator_
        # y_pred_log = best_model.predict(X_test)
        # y_pred_KNN = np.expm1(y_pred_log)

        # KNN fit
        KNN_pipeline.fit(X_train, y_train_log)

        KNN_pred_log = KNN_pipeline.predict(X_test)
        y_pred_KNN = np.expm1(KNN_pred_log)

        # Ridge fit
        Ridge_pipeline.fit(X_train, y_train_log)

        ridge_pred_log = Ridge_pipeline.predict(X_test)
        y_pred_ridge = np.expm1(ridge_pred_log)

        # RandomForest fit
        rf_pipeline.fit(X_train, y_train_log)

        rf_pred_log = rf_pipeline.predict(X_test)
        y_pred_rf = np.expm1(rf_pred_log)

        # Add Ridge and KNN
        y_pred = 0.3 * y_pred_KNN + 0.3 * y_pred_rf + 0.3 * y_pred_ridge

        # calculate MAE
        print("Ridge MAE:", mean_absolute_error(y_test, y_pred_ridge))
        print("KNN MAE:", mean_absolute_error(y_test, y_pred_KNN))
        print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
        print("Blend MAE:", mean_absolute_error(y_test, y_pred))
        # mae_main = mean_absolute_error(y_test, y_pred)
        # print(f"Original MAE: {mae_main:.4f} meters")

    # Show the first 5 results
    print("\n--- Final Distance Outputs ---")
    for i in range(5):
        print(f"Image {i}: Predicted Distance = {y_pred[i]:.4f} meters")

    # # --- THE SUBMISSION CODE GOES HERE ---
    # print("[INFO]: Generating Kaggle submission...")

    # test_images_raw = load_test_dataset(config)
    # test_images = np.array(test_images_raw)

    # processed_test_images = []

    # for img in test_images:
    #     img_2d = img.reshape(h, w)
    #     combined = img_2d.flatten()
    #     processed_test_images.append(combined)

    # test_images_processed = np.array(processed_test_images)

    # # predict on the test set
    # y_kaggle_log = best_model.predict(test_images_processed)
    # y_kaggle_final = np.expm1(y_kaggle_log)

    # # ... rest of the code I gave you ...
    # submission_df = pd.DataFrame({
    # "ID": [f"{i:03d}" for i in range(len(y_kaggle_final))],
    # "Distance": y_kaggle_final
    # })

    # submission_df.to_csv("KNN_Pipeline_GridSearch_submission.csv", index=False)
    
    # print(f"[SUCCESS]: Created submission with {len(submission_df)} rows!")

    # # 🔍 Error analysis by distance bins
    # errors = np.abs(y_pred - y_test)

    # bins = [0, 1.2, 1.5, 1.8, 2.2, 10]
    # names = ["very close", "close", "middle", "far", "very far"]

    

    # for low, high, name in zip(bins[:-1], bins[1:], names):
    #     mask = (y_test >= low) & (y_test < high)
    
    #     if mask.sum() > 0:  # avoid empty bins
    #         print(f"{name}: count={mask.sum()}, MAE={errors[mask].mean():.4f}")

    # worst_idx = np.argsort(errors)[-10:]

    # print("\nWorst predictions:")
    # for i in worst_idx:
    #     print(f"Idx {i}: Pred={y_pred[i]:.3f}, True={y_test[i]:.3f}, Error={errors[i]:.3f}")

    # for i in worst_idx:
    #     print(f"ID {id_test[i]}: Pred={y_pred[i]:.3f}, True={y_test[i]:.3f}")

    # # Plotting errors
    # img_id = id_test[i]
    # img_path = config["data_dir"] / "train_images" / f"{img_id}.png"

    # for i in worst_idx:
    #     img_id = id_test[i]
    #     img_path = config["data_dir"] / "train_images" / f"{img_id}.png"

    #     img = Image.open(img_path)

    #     plt.imshow(img, cmap="gray")
    #     plt.title(f"ID={img_id}, Pred={y_pred[i]:.2f}, True={y_test[i]:.2f}")
    #     plt.axis("off")
    #     plt.show()

    