import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("house_data5.csv")

# Features and target
X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Prediction
prediction = model.predict([[2100, 4, 3, 2]])

print("Predicted House Price:", prediction[0])
print("Model Accuracy:", accuracy)
