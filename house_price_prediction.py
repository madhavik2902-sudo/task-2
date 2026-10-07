import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# Load dataset
df = pd.read_csv("train copy.csv")

print("Dataset loaded successfully!")


# Select features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "YearBuilt",
    "Neighborhood"
]

target = "SalePrice"


# Input and target
X = df[features]
y = df[target]


# Numerical columns
numeric_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "YearBuilt"
]


# Categorical column
categorical_features = [
    "Neighborhood"
]


# Preprocessing for numerical columns
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# Preprocessing for location
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


print("\nPreprocessing setup completed!")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nData splitting completed!")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.linear_model import LinearRegression

# Create Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")
# Predict prices for testing data
y_pred = model.predict(X_test)

print("\nPredictions completed!")

print("\nFirst 10 Predictions:")
print(y_pred[:10])
# Compare actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error:", mae)

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()