import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample house data
data = {
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Price": [200000, 250000, 300000, 350000, 400000]
}

# Create dataframe
df = pd.DataFrame(data)

# Features and target
X = df[["Area"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[1700]])

print("Predicted House Price:", prediction[0])

# Plot points
plt.scatter(X, y)

# Regression line
plt.plot(X, model.predict(X))

# Labels
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")

# Show graph
plt.show()
