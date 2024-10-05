import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from flask import Flask, request, jsonify

# Load the dataset
url = "https://storage.googleapis.com/kagglesdsdata/datasets/752131/1621146/Unemployment%20in%20India.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241002%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241002T124609Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=4025cdb18a43b4a2412d8ca67f80ed8e29f1d0d458d1360fd6e1eee632e33b16bff0d4ed962bcfb03d0d77c036c7b64f54dad34f30ba4842157282121e38497732f94cb217724269a17606d77f703ba9ee8623bff6a9802f2bfd979db32f03234775efa1b0af501512d4db2c55053ddc7b06dc93065d44fcfc48f25e98a932129fb314692ae6adf9e7405fff5902e52f468c3c23e78119e98ee5c7b996a77c1b5c73acf76aba61388b0c4209234d8c3a205b7ca7398c89f979c94b09182605ee34db0701ebf87842d597444af2089420a583c0912e1c11960421cc56090ac6ee41e122c61c2b1e33e528b52edf98e8f7c355b12c132178e8748adb40e6bf18e7"
data = pd.read_csv(url)

# Clean column names
data.columns = data.columns.str.strip()
print(data.columns)

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Summary statistics
print(data.describe())

# Convert date to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Features and target variable
X = data[['Year', 'Month']]
y = data['Estimated Unemployment Rate (%)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {'fit_intercept': [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_
print(f'Best parameters: {grid.best_params_}')

# Save the model
joblib.dump(model, 'unemployment_rate.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Data Visualization
# Unemployment rate distribution
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title('Distribution of Unemployment Rate')
plt.show()

# Unemployment rate over time
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data)
plt.title('Unemployment Rate Over Time')
plt.xticks(rotation=45)
plt.show()

# Unemployment rate by region
plt.figure(figsize=(12, 8))
sns.boxplot(x='Region', y='Estimated Unemployment Rate (%)', data=data)
plt.title('Unemployment Rate by Region')
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

