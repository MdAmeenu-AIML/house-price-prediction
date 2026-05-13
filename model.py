import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("house_data.csv")

# Features and target
X = df[["Area"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[1700]])

print("Predicted House Price:", prediction[0])

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction V3")

plt.show()
