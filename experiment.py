import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240522154015/Salary_Data[1].csv'
df = pd.read_csv(url)

# Drop rows with missing values
df_cleaned = df.dropna()

# Define features and target variable
X = df_cleaned.drop('Salary', axis=1)
y = df_cleaned['Salary']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (scaling numerical features and one-hot encoding categorical features)
numeric_features = ['Age', 'Years of Experience']
categorical_features = ['Gender', 'Education Level', 'Job Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Preprocess the training and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Linear Regression 
model_lr = LinearRegression()
model_lr.fit(X_train_processed, y_train)

# Gradient Boosting Regression
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_gb.fit(X_train_processed, y_train)

# K-Nearest Neighbors Regression
model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train_processed, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_test_processed)
y_pred_gb = model_gb.predict(X_test_processed)
y_pred_knn = model_knn.predict(X_test_processed)

# Model evaluation
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\nGradient Boosting:")
print("MSE:", mean_squared_error(y_test, y_pred_gb))
print("R2 Score:", r2_score(y_test, y_pred_gb))

print("\nK-NN:")
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print("R2 Score:", r2_score(y_test, y_pred_knn))

# Predict salary for new data
new_data = pd.DataFrame({
    'Age': [30],
    'Years of Experience': [5],
    'Gender': ['Male'],
    'Education Level': ['Bachelor\'s'],
    'Job Title': ['Software Engineer']
})

new_data_processed = preprocessor.transform(new_data)

predicted_value_lr = model_lr.predict(new_data_processed)
predicted_value_gb = model_gb.predict(new_data_processed)
predicted_value_knn = model_knn.predict(new_data_processed)

print("\nPredicted Salary (Linear Regression):", predicted_value_lr[0])
print("\nPredicted Salary (Gradient Boosting):", predicted_value_gb[0])
print("\nPredicted Salary (K-NN):", predicted_value_knn[0])