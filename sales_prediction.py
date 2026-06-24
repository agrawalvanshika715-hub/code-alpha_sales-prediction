import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# First 5 rows
print("Dataset Preview:")
print(df.head())

# Dataset info
print("\nDataset Information:")
print(df.info())

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)

print("\nModel Accuracy (R2 Score):", score)

# Future Sales Prediction
future_sales = model.predict([[200, 40, 50]])

print("\nPredicted Sales for TV=200, Radio=40, Newspaper=50:")
print(future_sales[0])