import sys
print(sys.executable)
from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import datasets, svm, preprocessing, linear_model
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


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

    # normal Train Test Split on 20% of Data
    X_train, X_test, y_train, y_test = train_test_split(images, distances, test_size=0.2, random_state=42)

    # Creating bins for stratisfied K-Folds
    y_bins = np.digitize(y_train, bins=np.linspace(y_train.min(), y_train.max(), 10))

    # Set up Stratified KFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Log transform distance labels
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Initializing Pipeline KNN
    KNN_pipeline = make_pipeline(
                        preprocessing.RobustScaler(), 
                        PCA(n_components=150), 
                        KNeighborsRegressor(n_neighbors=5, weights='distance'))
    
    # set up grid search
    param_distributions = {
        'pca__n_components': [70, 75, 80, 85],
        'kneighborsregressor__n_neighbors' : [2, 3, 4, 5, 6],
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__p':[1]
    }

    # Fit and Predict
    random_search = RandomizedSearchCV(
        KNN_pipeline, 
        param_distributions,
        n_iter=80, 
        cv=skf.split(X_train, y_bins),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train_log)

    # Use the best model
    best_model = random_search.best_estimator_
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Show the first 5 results
    print("\n--- Final Distance Outputs ---")
    for i in range(5):
        print(f"Image {i}: Predicted Distance = {y_pred[i]:.4f} meters")
    print(f"Mean Absolute Error: {mae:.4f} meters")

    # --- THE SUBMISSION CODE GOES HERE ---
    print("[INFO]: Generating Kaggle submission...")
    test_images_raw = load_test_dataset(config)
    test_images = np.array(test_images_raw)
    test_images_flat = test_images.reshape(len(test_images), -1)

    # predict on the test set
    y_kaggle_log = best_model.predict(test_images_flat)
    y_kaggle_final = np.expm1(y_kaggle_log)

    # ... rest of the code I gave you ...
    submission_df = pd.DataFrame({
    'ID': np.arange(len(y_kaggle_final)), 
    'Distance': y_kaggle_final
    })
    
    submission_df.to_csv("KNN_Pipeline_GridSearch_submission.csv", index=False)
    print(f"[SUCCESS]: Created submission with {len(submission_df)} rows!")

    