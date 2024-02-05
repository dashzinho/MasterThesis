import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
from tabulate import tabulate
import os

###################
# Iinitialization #
###################

# Load data file
file = "../data.csv"
df = pd.read_csv(file, index_col=0)

# Define the maturities
maturities = [3, 6, 12, 24, 36, 60, 72, 120]

# Define the number of forecast periods
forecast_periods = int(input("Enter the number of forecast periods: "))

# Create a new folder if it doesn't exist
folder_name = f'../Figures/Results/GPR/{forecast_periods} months'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Create a list to store the results
results = []
predicted_yields = []
small_period_values = []


########
# Loop #
########

for maturity in maturities:

    print(f"Processing Maturity: {maturity}")

    # Extract the observed yields for the current maturity
    observed_yields = df[str(maturity)].values

    # Prepare the data for training
    X, y = [], []
    for i in range(forecast_periods, len(observed_yields)):
        X.append(observed_yields[i - forecast_periods:i])
        y.append(observed_yields[i])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=forecast_periods, shuffle=False)

    # Initialize a GPR model with a kernel
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5)) + WhiteKernel(noise_level=1.7e-1)
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=forecast_periods)

    # Fit the GPR model
    gpr_model.fit(X_train, y_train)

    # Predict the next 'forecast_periods' values
    y_pred, sigma = gpr_model.predict(X_test, return_std=True)

    # Get the forecasted dates
    forecasted_dates = df.index[-forecast_periods:]

    # Calculate residuals
    residuals = y_test - y_pred

    # Create a DataFrame for the test and forecasted yields
    forecasted_yield_curve = pd.DataFrame({
        'Maturity': maturity,
        'Actual': y_test,
        'GPR Forecast': y_pred,
        'Residuals': residuals
    }, index=forecasted_dates)

    # Append the predicted yields for visualization
    predicted_yields.append(y_pred)
    small_period_values.append(forecasted_yield_curve)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Maturity": maturity,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

    print('----------------------------------------------------------')
    print(forecasted_yield_curve)
    print('----------------------------------------------------------')

    if forecast_periods > 3:

        # Plot the actual and predicted yield curve for forecasted periods
        plt.figure(figsize=(10, 6))
        plt.plot(forecasted_dates, y_test, marker='o', label='Actual', color='C3')
        plt.plot(forecasted_dates, y_pred, label='GPR Forecast', linestyle='--', marker='o', color='C0')
        plt.title(f'Maturity {maturity} Yields Forecast (Last {forecast_periods} Months)')
        plt.xlabel('Time')
        plt.ylabel('Yield')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save the figure in the new folder
        plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_{maturity}mat_forecast.png'), bbox_inches='tight')
        plt.show()        


        # Plot residuals
        plt.figure(figsize=(10, 6))
        plt.plot(forecasted_dates, residuals, label='Residuals', marker='o', linestyle='--', color='C0')
        plt.axhline(0, color='C3', linestyle='-', linewidth=1)
        plt.title(f'Maturity {maturity} Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residuals')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save the figure in the new folder
        plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_{maturity}mat_residuals.png'), bbox_inches='tight')
        plt.show()

########################
# Tables,excels,graphs #
########################

# Concat for better visualization
small_period_values = pd.concat(small_period_values,ignore_index=True)

# save the file as excel in the new folder
excel_file_path = os.path.join(folder_name, f'{forecast_periods}m_yieldcurve_data.xlsx')
small_period_values.to_excel(excel_file_path, index=False)

# Convert predicted_yields to a DataFrame
predicted_df = pd.DataFrame(data=np.array(predicted_yields).T, columns=maturities, index=df.index[-forecast_periods:])

# Get the last forecasted date
last_date = predicted_df.index[-1]

# Extract actual yield values for the last date
actual_yields = df.loc[last_date, [str(maturity) for maturity in maturities]]

# Plot both actual and predicted yield curves for the last forecasted date
plt.figure(figsize=(10, 6))
plt.plot(predicted_df.columns, predicted_df.loc[last_date], marker='o', label=f'GPR Forecasted Yield Curve on {last_date}', color='C0')
plt.plot(actual_yields.index.astype(int), actual_yields.values, marker='o', label=f'Actual Yield Curve on {last_date}', color='C3')
plt.title(f'Actual vs Forecasted Yield Curve for {last_date}')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Save the figure in the new folder
plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_yieldcurve.png'), bbox_inches='tight') 
plt.show()

# Create a table with evaluation metrics
results_df = pd.DataFrame(results)

# Round the numbers in the DataFrame
results_df_rounded = results_df.round(5)

# Print evaluation metrics in a formatted table
eval_table = tabulate(results_df_rounded, headers='keys', tablefmt='latex', showindex=False)
print("Evaluation Metrics:")
print(eval_table)

# Define the file path
eval_file_path = os.path.join(folder_name, 'eval_metrics.txt')

with open(eval_file_path, 'w') as file:
    file.write("Evaluation Metrics:\n")
    file.write(eval_table)


def summary_table(dfx):
    summary_data = {
        "Mean": dfx.mean(),
        "Std": dfx.std(),
        "Min": dfx.min(),
        "Max": dfx.max(),
    }
    summary_df = pd.DataFrame(summary_data)

    # Round the numbers in the summary DataFrame
    summary_df_rounded = summary_df.round(5)

    return summary_df_rounded

summary_df = summary_table(results_df)

# Print summary metrics in a formatted table
summary_table_formated = tabulate(summary_df, headers='keys', tablefmt='latex', showindex=True)
print("\nSummary Metrics:")
print(summary_table_formated)

# Define the file path
summary_file_path = os.path.join(folder_name, 'summary_metrics.txt')

with open(summary_file_path, 'w') as file:
    file.write("Summary Metrics:\n")
    file.write(summary_table_formated)