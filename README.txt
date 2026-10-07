HOUSE PRICE PREDICTION USING LINEAR REGRESSION

Internship Task 2

1. PROJECT TITLE
House Price Prediction using Linear Regression

2. OBJECTIVE
The objective of this project is to develop a Linear Regression
model to predict house prices based on different features such as
house quality, living area, number of bedrooms, bathrooms, garage
capacity, year built, basement area, total rooms, and location.

3. DATASET
Dataset: House Prices - Advanced Regression Techniques
Source: Kaggle

Dataset file used:
train.csv

Target variable:
SalePrice

4. FEATURES USED
The following features were used for prediction:

- OverallQual
- GrLivArea
- GarageCars
- TotalBsmtSF
- FullBath
- BedroomAbvGr
- TotRmsAbvGrd
- YearBuilt
- Neighborhood

5. TECHNOLOGIES USED
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

6. DATA PREPROCESSING
The dataset was preprocessed before training the model.

- Missing numerical values were replaced using the median.
- Missing categorical values were replaced using the most frequent value.
- The Neighborhood categorical feature was converted into numerical
  form using One-Hot Encoding.

7. MODEL USED
Linear Regression was used to train the house price prediction model.

The dataset was divided into:
- 80% Training Data
- 20% Testing Data

Training data: 1168 rows
Testing data: 292 rows

8. MODEL EVALUATION

Mean Absolute Error (MAE):
22341.74

Mean Squared Error (MSE):
1304619311.81

Root Mean Squared Error (RMSE):
36119.51

R2 Score:
0.8299

The R2 score indicates that the model explains approximately
82.99% of the variation in house prices in the test dataset.

9. OUTPUT
The model generates predicted house prices for the test data.

An Actual vs Predicted House Prices graph was created to compare
the actual house prices with the predicted prices.

10. PROJECT FILES

house_price_prediction.py
    Main Python program for house price prediction.

train.csv
    Kaggle dataset used for training and testing.

actual_vs_predicted.png
    Graph comparing actual and predicted house prices.

README.txt
    Project documentation.

11. CONCLUSION
The Linear Regression model was successfully developed to predict
house prices using selected house features. The model was trained
and tested successfully, and its performance was evaluated using
MAE, MSE, RMSE, and R2 Score.