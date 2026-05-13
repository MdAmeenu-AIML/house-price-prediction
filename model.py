import pandas as pd
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

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for 1700 sq ft
prediction = model.predict([[1700]])

print("Predicted House Price:", prediction[0])
