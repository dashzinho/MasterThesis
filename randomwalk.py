import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate
import os

##################
# Initialization #
##################

# Load file
file = "../data.csv"
df = pd.read_csv(file, index_col=0)

# Random Noise calculation
random_noise = df.diff()
random_noise = random_noise.dropna().abs().mean().mean()
print('Random Noise:',random_noise)

# Define the number of forecast months
forecast_periods = int(input("Enter the number of forecast periods: "))

# Create a new folder if it doesn't exist
folder_name = f'../Figures/Results/RW/{forecast_periods} months'
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

    # Get the most recent observed yield
    last_observed_yield = observed_yields.iloc[-forecast_periods - 1]

    # Initialize the forecasted yield with the last observed yield
    forecasted_yields = [last_observed_yield]

    # Generate random noise for each forecast period and update the forecasted yield
    for i in range(forecast_periods):
        next_forecast = forecasted_yields[-1] + np.random.normal(0, random_noise)
        forecasted_yields.append(next_forecast)

    # Remove the initial observed yield from the forecasted list
    forecasted_yields = forecasted_yields[1:]

    # Calculate residuals
    residuals = observed_yields[-forecast_periods:] - forecasted_yields

    # Create a DataFrame for the forecasted yields
    forecasted_yield_curve = pd.DataFrame({
        'Maturity': maturity,
        'Actual': observed_yields[-forecast_periods:],
        'Random Walk Forecast': forecasted_yields,
        'Residuals': residuals
    })

    # Append the predicted yields for visualization
    predicted_yields.append(forecasted_yields)
    small_period_values.append(forecasted_yield_curve)

    # Calculate evaluation metrics
    mse = mean_squared_error(observed_yields[-forecast_periods:], forecasted_yields)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(observed_yields[-forecast_periods:], forecasted_yields)

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
        plt.plot(forecasted_yield_curve.index, forecasted_yield_curve['Actual'], label='Actual', marker='o', color='C3')
        plt.plot(forecasted_yield_curve.index, forecasted_yield_curve['Random Walk Forecast'], label='Random Walk Forecast', marker='o', linestyle='--', color='C0')
        plt.title(f'Maturity {maturity} Yields Forecast (Last {forecast_periods} Months)')
        plt.xlabel('Date')
        plt.ylabel('Yield')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save the figure in the new folder
        plt.savefig(os.path.join(folder_name, f'{forecast_periods}m_{maturity}mat_forecast.png'), bbox_inches='tight')
        plt.show()        

        # Plot the residuals
        plt.figure(figsize=(10, 6))
        plt.plot(residuals.index, residuals, label='Residuals', marker='o', linestyle='--', color='C0')
        plt.axhline(y=0, color='C3', linestyle='-', label='Zero Residuals')
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
plt.plot(predicted_df.columns, predicted_df.loc[last_date], marker='o', label=f'RW Forecasted Yield Curve on {last_date}', color='C0')
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

# Open the file in write mode and print the formated summary table into it
with open(summary_file_path, 'w') as file:
    file.write("Summary Metrics:\n")
    file.write(summary_table_formated)