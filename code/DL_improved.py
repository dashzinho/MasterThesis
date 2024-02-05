import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import os
import matplotlib.pyplot as plt
from tabulate import tabulate


########
# Data #
########

# Load file
file = "../data.csv"
df = pd.read_csv(file, index_col=0)


##################
# Method-Related #
##################

def load2(x, lam):
    return (1 - np.exp(-lam * x)) / (lam * x)

def load3(x, lam):
    return ((1 - np.exp(-lam * x)) / (lam * x)) - np.exp(-lam * x)


maturity_cols = np.array([3, 6, 12, 24, 36, 60, 72, 120])
betas_cols = ["b1", "b2", "b3"]


# Compute betas and residuals function
def compute_betas_resid(df, lam, alpha=0.29):
    X = np.zeros((len(maturity_cols), 2)) # zero matrix 8x2
    X[:, 0] = load2(maturity_cols, lam) # b2 features
    X[:, 1] = load3(maturity_cols, lam) # b3 features
    X = sm.add_constant(X) # column of ones

    betas = np.zeros((len(df), 3))
    ridge_residuals = np.zeros((len(df), 8))

    for i in range(0, len(df)): #loop through each df row
        ridge_model = Ridge(alpha=alpha, fit_intercept=False) # modelisation
        ridge_model.fit(X, df.iloc[i]) # fits model to the X features and df target values
        betas[i, :3] = ridge_model.coef_ # store betas
        ridge_residuals[i, :] = df.iloc[i] - ridge_model.predict(X) # store residuals

    betas = pd.DataFrame(betas, columns=betas_cols)
    ridge_residuals = pd.DataFrame(ridge_residuals, columns=[str(i) for i in maturity_cols])

    betas.index = df.index
    ridge_residuals.index = df.index

    return betas, ridge_residuals



#################
# Initilization #
#################

# function to compute back yields
def compute_yield(row, lam, maturity):
    beta1, beta2, beta3 = row['b1'], row['b2'], row['b3']

    Y_t = beta1 + beta2 * (1 - np.exp(-lam * maturity)) / (lam * maturity) + beta3 * (
                (1 - np.exp(-lam * maturity)) / (lam * maturity)) - np.exp(-lam * maturity)

    return Y_t

maturities = [3, 6, 12, 24, 36, 60, 72, 120]
forecast_periods = int(input("Enter the number of forecast periods: "))

# Create a new folder if it doesn't exist
folder_name = f'../Figures/Results/DL/{forecast_periods} months'
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
    best_lam = None
    max_loading = -np.inf

    for current_lam in np.arange(0.01, 0.1, 0.01):

        betas, _ = compute_betas_resid(df[:-forecast_periods], current_lam)
        loading_beta3 = abs(betas['b3']).max()

        if loading_beta3 > max_loading:
            max_loading = loading_beta3
            best_lam = current_lam
    
    betas, _ = compute_betas_resid(df, best_lam)

    train, test = betas[0:-forecast_periods], betas[-forecast_periods:]
    # Define the lag order for the VAR model
    VAR_model = VAR(train)
    lags = VAR_model.select_order()
    lag_order = lags.aic

    # Fit the VAR model
    VAR_model_fit = VAR_model.fit(lag_order)

    # Forecast the next 'forecast_periods' periods
    forecast = VAR_model_fit.forecast(train.values[-lag_order:], steps=forecast_periods)
    forecasted_betas = pd.DataFrame(forecast, columns=betas_cols)

    # Compute yields using the forecasted betas
    forecasted_yields = []
    for index, row in forecasted_betas.iterrows():
        yield_value = compute_yield(row, best_lam, maturity)
        forecasted_yields.append(yield_value)

    observed_yields = df[str(maturity)]
    observed_yields = observed_yields[-forecast_periods:]

    residuals = observed_yields.values - forecasted_yields

    # Create a DataFrame for the test and forecasted yields
    forecasted_yield_curve = pd.DataFrame({
        'Maturity': maturity,
        'Actual': observed_yields.values,
        'DL Forecast': forecasted_yields,
        'Residuals': residuals
    })

    # Append the predicted yields for visualization
    predicted_yields.append(forecasted_yields)
    small_period_values.append(forecasted_yield_curve)

    forecasted_yield_curve.index = observed_yields.index

    mse = mean_squared_error(observed_yields, forecasted_yields)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(observed_yields, forecasted_yields)

    results.append({
        "Maturity": maturity,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    })


    print('----------------------------------------------------------')
    print(forecasted_yield_curve)
    print('----------------------------------------------------------')

    if forecast_periods > 3:

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(observed_yields.index, observed_yields.values, label='Actual', marker='o', color='C3')
        plt.plot(observed_yields.index, forecasted_yields, label='Diebold-Li Forecast', linestyle='--', marker='o', color='C0')
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
        plt.figure(figsize=(10, 6))
        plt.plot(observed_yields.index, residuals, label='Residuals', marker='o', color='C0')
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


predicted_df = pd.DataFrame(data=np.array(predicted_yields).T, columns=maturities, index=df.index[-forecast_periods:])

last_date = predicted_df.index[-1]

# Extract actual yield values for the last date
actual_yields = df.loc[last_date, [str(maturity) for maturity in maturities]]

# Plot both actual and predicted yield curves for the last forecasted date
plt.figure(figsize=(10, 6))
plt.plot(predicted_df.columns, predicted_df.loc[last_date], marker='o', label=f'DL Forecasted Yield Curve on {last_date}', color='C0')
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



results_df = pd.DataFrame(results)
results_df_rounded = results_df.round(5)

# Print evaluation metrics in a formatted table
eval_table = tabulate(results_df_rounded, headers='keys', tablefmt='latex', showindex=False)
print("Evaluation Metrics:")
print(eval_table)


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

    summary_df_rounded = summary_df.round(5)

    return summary_df_rounded

summary_df = summary_table(results_df)

# Print summary metrics in a formatted table
summary_table_formated = tabulate(summary_df, headers='keys', tablefmt='latex', showindex=True)
print("\nSummary Metrics:")
print(summary_table_formated)

summary_file_path = os.path.join(folder_name, 'summary_metrics.txt')

with open(summary_file_path, 'w') as file:
    file.write("Summary Metrics:\n")
    file.write(summary_table_formated)
