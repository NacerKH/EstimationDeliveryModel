

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
import numpy as np
from tqdm import tqdm  # Import tqdm

# Load the CSV file into a DataFrame
df = pd.read_csv("data/dataset.csv")

# Calculate the percentage of missing values in each column
missing_percentages = df.isnull().mean() * 100

# Identify columns with missing values exceeding 5%
high_missing_columns = missing_percentages[missing_percentages > 5].index

# Drop rows with missing values exceeding 5%
df = df.dropna(subset=high_missing_columns)

# Handle remaining missing values (not exceeding 5%) using mode imputation
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Export the DataFrame to a new CSV file with imputed values
df.to_csv("imputed_datadelivery.csv", index=False)
di = pd.read_csv("imputed_datadelivery.csv")
print(di.isna().values.any())

# Select categorical columns in the DataFrame
categorical_cols = di.columns[di.dtypes==object].tolist()

# Instantiate LabelEncoder
encoder = LabelEncoder()

# Columns to be encoded
columns_to_encode = ['created_at', 'actual_delivery_time', 'store_id', 'store_primary_category']

# Apply encoding to all specified columns in the DataFrame di
di[columns_to_encode] = di[columns_to_encode].apply(lambda col: encoder.fit_transform(col.astype(str)))

print(di.head(5))

# Normalize features
X = di.drop('actual_delivery_time', axis=1)
y = di[['actual_delivery_time']]

sc = MinMaxScaler()
X_sc = sc.fit_transform(X)
X = pd.DataFrame(X_sc, columns=X.columns)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print(X_train.shape)
print(X_test.shape)

# Instantiate the regression models
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=10),
    "SVR": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=0),
    "Ridge": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Train and evaluate each model
best_model = None
best_score = -float('inf')

# Use tqdm to create a progress bar
for name, model in tqdm(models.items(), desc="Training Progress"):
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name}:")
    print("R² Score:", r2)
    print("Mean Squared Error:", mse)
    if r2 > best_score:
        best_score = r2
        best_model = name

# Print the best model
print("\033[92m" + "Best Model:" + "\033[0m", best_model)  # Print Best Model in green
print("\033[92m" + "Best R² Score:" + "\033[0m", best_score)  # Print Best R² Score in green
