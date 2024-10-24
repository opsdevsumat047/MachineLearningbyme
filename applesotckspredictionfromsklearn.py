import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy.interpolate import UnivariateSpline

def fetch_apple_stock_data(start_date, end_date):
    apple_data = yf.download('AAPL', start=start_date, end=end_date)
    X = np.array([d.toordinal() for d in apple_data.index])
    y = apple_data['Adj Close'].values
    return X, y

# Fetch data
start_date = '2023-01-01'
end_date = '2023-06-06'
X_tmp, y_tmp = fetch_apple_stock_data(start_date, end_date)

# Fit a spline (you can adjust the s parameter for smoothing)
spline = UnivariateSpline(X_tmp, y_tmp, s=1)  # s controls the smoothness

# Generate predictions
X_range = np.linspace(X_tmp.min(), X_tmp.max(), 100)
predicted_y = spline(X_range)

# Prepare dates for plotting
dates = [datetime.date.fromordinal(int(x)) for x in X_tmp]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(dates, y_tmp, label='Actual Stock Prices', color='blue')
plt.plot([datetime.date.fromordinal(int(x)) for x in X_range], predicted_y, label='Predicted Stock Prices (Spline)', color='red', linestyle='--')
plt.title("Apple Stock Prices: Actual vs. Predicted (Spline Regression)")
plt.xlabel("Date")
plt.ylabel("Stock Price (Adjusted Close in $)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
