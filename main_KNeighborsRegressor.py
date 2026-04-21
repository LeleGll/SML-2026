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
from sklearn.model_selection import cross_val_score
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

    # # Convert distances into categories
    # distances_binned = pd.cut(distances, bins=10, labels=False)

    # normal Train Test Split on 20% of Data
    X_train, X_test, y_train, y_test = train_test_split(images, distances, test_size=0.2, random_state=42)

    # # set up linear Regression
    # reg = linear_model.LinearRegression()

    # # set up pipeline SVC
    # clf = make_pipeline(preprocessing.StandardScaler(), 
    #                     PCA(n_components=0.95), 
    #                     svm.SVC(kernel='rbf', C=10))
    
    # set up pipeline linear Regression
    # reg_clf = make_pipeline(
    #                     preprocessing.StandardScaler(), 
    #                     PCA(n_components=100), 
    #                     Lasso(alpha=.000001))

    # Scaling data for KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initializing KNN
    KNN = KNeighborsRegressor(n_neighbors=5, weights='distance')

    # Fit and Predict
    KNN.fit(X_scaled, y_train)
    y_pred = KNN.predict(X_test_scaled)

    # # Stratified K-Fold Cross Validation on rest of the data
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(reg_clf, X_train, y_train, cv=skf, scoring='r2')
    # print(f"Model Quality (R2 Score): {cv_scores.mean():.2f}")

    # # Train the model
    # reg_clf.fit(X_train, y_train)

    # # output distance
    # final_predictions = reg_clf.predict(X_test)

    # calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Show the first 5 results
    print("\n--- Final Distance Outputs ---")
    for i in range(5):
        print(f"Image {i}: Predicted Distance = {y_pred[i]:.4f} meters")
    print(f"Mean Absolute Error: {mae:.4f} meters")

    # # Normalization of data
    # scaler = preprocessing.StandardScaler().fit(X_train)
    # # print(scaler.mean_)

    # # scale data
    # X_scaled = scaler.transform(X_train)

    # Cross Validation on Training set

    # for train, test in skf.split(X,y):
    #     print('train - {}   |   test - {}'.format(
    #         np.bincount(y[train]), np.bincount(y[test])))
    
    
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # print(clf.score(X_test, y_test))

    # loading and plotting distance labels
    # df = pd.read_csv(file_path)
    # print(df.to_string())

    # plotting data
    # plt.figure(figsize=(10,6))
    # plt.hist(df['distance'], bins=30, edgecolor='black')
    # plt.title('Distribution of Distance Labels')
    # plt.xlabel('Distrance Value')
    # plt.ylabel('Number of Samples (Frequency)')

    # plt.show()

    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(test_pred)