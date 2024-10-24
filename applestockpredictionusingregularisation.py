import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import matplotlib.dates as mdates

def fetch_apple_stock_data(start_date, end_date):
    apple_data = yf.download('AAPL', start=start_date, end=end_date)
    X = np.array([d.toordinal() for d in apple_data.index]).reshape(-1, 1)
    y = apple_data['Adj Close'].values
    return X, y

def predict(X, w, b):
    return np.dot(X, w) + b

def polynomial_features(X, degree):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    return X_poly

def compute_cost_polynomial_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    cost = np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost

# Fetch Apple stock data
start_date = '2023-01-01'
end_date = '2023-06-06'
X_tmp, y_tmp = fetch_apple_stock_data(start_date, end_date)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_tmp)
y_scaled = scaler_y.fit_transform(y_tmp.reshape(-1, 1)).flatten()

# Set the degree of the polynomial (consider a lower degree)
degree = 6
X_poly_tmp = polynomial_features(X_scaled, degree)

# Initialize model parameters
w_tmp = np.random.rand(X_poly_tmp.shape[1]) * 0.1  # Smaller initialization
b_tmp = 0.0
lambda_tmp = 0.4

# Compute cost
cost_tmp = compute_cost_polynomial_reg(X_poly_tmp, y_scaled, w_tmp, b_tmp, lambda_tmp)
predicted_y_scaled = predict(X_poly_tmp, w_tmp, b_tmp)

# Inverse transform the predicted values
predicted_y = scaler_y.inverse_transform(predicted_y_scaled.reshape(-1, 1)).flatten()

# Print the predicted values to check for anomalies
print("Predicted y values (in $):", predicted_y)

# Prepare dates for plotting
dates = [datetime.date.fromordinal(int(x)) for x in X_tmp]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(dates, y_tmp, label='Actual Stock Prices', color='blue')
plt.plot(dates, predicted_y, label='Predicted Stock Prices (Polynomial Regression)', color='red', linestyle='--')
plt.title("Apple Stock Prices: Actual vs. Predicted (Polynomial Degree = 6)")
plt.xlabel("Date")
plt.ylabel("Stock Price (Adjusted Close in $)")  # Updated label
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("Regularized cost with polynomial regression for Apple stock:", cost_tmp)
