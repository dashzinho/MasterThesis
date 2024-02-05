import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tabulate import tabulate
import os

#################
# Initilization #
#################

# Load data file
file = "../data.csv"
df = pd.read_csv(file, index_col=0)

# Define the number of forecast months
forecast_periods = int(input("Enter the number of forecast periods: "))

# Create a new folder if it doesn't exist
folder_name = f'../Figures/Results/ExpSmooth/{forecast_periods} months'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define maturities
maturities = [3, 6, 12, 24, 36, 60, 72, 120]

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
    observed_yields = df[str(maturity)]

    # Split the data into training and testing sets
    train = observed_yields[:-forecast_periods]
    test = observed_yields[-forecast_periods:]

    # Apply Holt-Winters exponential smoothing to the training data
    model = ExponentialSmoothing(train, trend="add", seasonal="add", damped_trend=True, seasonal_periods=12)
    phi=0.8

    # Fit the model
    model_fit = model.fit(damping_trend=phi)

    # Forecast the next 'forecast_periods' values
    forecasted_yields = model_fit.forecast(steps=forecast_periods)

    # Calculate residuals
    residuals = test.values - forecasted_yields

    # Create a DataFrame for the test and forecasted yields
    forecasted_yield_curve = pd.DataFrame({
        'Maturity': maturity,
        'Actual': test.values,
        'ExpSmooth Forecast': forecasted_yields,
        'Residuals': residuals
    })
    
    # Append the predicted yields for visualization
    predicted_yields.append(forecasted_yields)
    small_period_values.append(forecasted_yield_curve)

    # Set the index to the date
    forecasted_yield_curve.index = test.index

    # Calculate evaluation metrics
    mse = mean_squared_error(test, forecasted_yields)
    rmse = sqrt(mse)
    mae = mean_absolute_error(test, forecasted_yields)

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

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(test.index, test.values, label='Actual', marker='o', color='C3')
        plt.plot(test.index, forecasted_yields, label='Exponential Smoothing Forecast', linestyle='--', marker='o', color='C0')
        plt.title(f'Maturity {maturity} Yields Forecast (Last {forecast_periods} Months)')
        plt.xlabel('Date')
        plt.ylabel('Yield')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save the figure in the new folder
        plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_{maturity}mat_forecast.png'), bbox_inches='tight') 
        plt.show()    

        # Plot residuals
        plt.figure(figsize=(10, 4))
        plt.plot(test.index, residuals, label='Residuals', marker='o', color='C0')
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
plt.plot(predicted_df.columns, predicted_df.loc[last_date], marker='o', label=f'ExpSmooth Forecasted Yield Curve on {last_date}', color='C0')
plt.plot(actual_yields.index.astype(int), actual_yields.values, marker='o', label=f'Actual Yield Curve on {last_date}', color='C3')
plt.title(f'Actual vs Forecasted Yield Curve for {last_date}')
plt.xlabel('Maturity')
plt.ylabel('Yield')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Save the figure in the new folder
plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_yieldcurve.png'), bbox_inches='tight')  # Save as PNG format
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
