import sys
print(sys.executable)
from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn import datasets, svm, preprocessing, linear_model
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from PIL import Image

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

    # setting up alphas
    alphas = np.logspace(-3, 3, 10)

    # Convert distances into categories
    distances_bins = np.digitize(distances, bins=[1.2, 1.5, 1.7, 1.9, 2.2])
    labels_df = pd.read_csv(config["data_dir"] / "train_labels.csv", dtype={"ID": str})
    ids = labels_df["ID"].values

    # normal Train Test Split on 20% of Data
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        images, 
        distances,
        ids, 
        test_size=0.2, 
        random_state=42,
        stratify=distances_bins)
    
    y_train_log = np.log1p(y_train)

    Ridge_pipeline = make_pipeline(
                        preprocessing.StandardScaler(), 
                        PCA(n_components=200), 
                        Ridge(alpha=1438.44988828766)
                    )
    # param_grid = {
    # "pca__n_components": [50, 80, 100, 120, 150, 200],
    # "ridge__alpha": np.logspace(-4, 4, 20)
    # }

    # ridge_search = GridSearchCV(
    #     Ridge_pipeline,
    #     param_grid,
    #     cv=5,
    #     scoring="neg_mean_absolute_error",
    #     n_jobs=-1
    #     )
    
    Ridge_pipeline.fit(X_train, y_train_log)
    # print("Best params:", ridge_search.best_params_)
    # print("Best CV MAE:", -ridge_search.best_score_)

    ridge_pred_log = Ridge_pipeline.predict(X_test)
    y_pred = np.expm1(ridge_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ridge MAE: {mae:.4f} meters")

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

    