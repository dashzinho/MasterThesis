import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os


# Load file
file = "../data.csv"
df = pd.read_csv(file, index_col=0)


############################################
# Diebold Li tables and data visualization #
############################################

# Create a new folder if it doesn't exist
folder_name = '../Figures/DL_replication'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#stuff
maturity_cols = np.array([3, 6, 12, 24, 36, 60, 72, 120])
betas_cols = ["b1", "b2", "b3"]
lam = 0.0609

load2 = lambda x: (1-np.exp(-lam*x)) / (lam*x)
load3 = lambda x: ((1-np.exp(-lam*x)) / (lam*x)) - np.exp(-lam*x)


###########
#  Loads  #
###########

timetomat = np.array([3, 6, 9, 12, 24, 36, 60, 72, 84, 120])
avg_maturity = np.mean(timetomat)

Z = np.zeros((len(timetomat), 3))
Z[:, 0] = np.ones(len(timetomat))
Z[:, 1] = (1 - np.exp(-lam * timetomat)) / (lam * timetomat)
Z[:, 2] = ((1 - np.exp(-lam * timetomat)) / (lam * timetomat)) - np.exp(-lam * timetomat)

plt.figure(figsize=(8, 6))
plt.plot(timetomat, Z[:, 0], label='Beta1')
plt.plot(timetomat, Z[:, 1], label='Beta2')
plt.plot(timetomat, Z[:, 2], label='Beta3')
plt.title(f'Factor Loadings for Diebold Li Model with Time Factor of {lam}')
plt.xlabel('Maturity (Months)')
plt.ylabel('Factor Loadings')
plt.ylim([0, 1.1])
plt.legend()
plt.show()


###########
# Table 1 #
###########

def summary_table(dfx):
    summary_data = {
        "Mean": dfx.mean(),
        "Std": dfx.std(),
        "Min": dfx.min(),
        "Max": dfx.max(),
    }
    summary_dfx = pd.DataFrame(summary_data)
    
    return summary_dfx

summary_df = summary_table(df)
print(summary_df)

excel_file_path = os.path.join(folder_name, 'Table1_replication.xlsx')
summary_df.to_excel(excel_file_path, index=False)

###########
# Table 2 #
###########

X = np.zeros((len(maturity_cols), 2))
X[:, 0] = load2(maturity_cols)
X[:, 1] = load3(maturity_cols)
X = sm.add_constant(X)

betas = np.zeros((len(df), 3))
residuals = np.zeros((len(df), 8))

for i in range(0, len(df)):
    model = sm.regression.linear_model.OLS(df.iloc[i], X)
    results = model.fit()
    betas[i, :3] = results.params
    residuals[i, :] = results.resid

betas = pd.DataFrame(betas, columns=betas_cols)
residuals = pd.DataFrame(residuals, columns=[str(i) for i in maturity_cols])

betas.index = df.index
residuals.index = df.index

summary_residuals = summary_table(residuals)
print(summary_residuals)

excel2_file_path = os.path.join(folder_name, 'Table2_replication.xlsx')
summary_residuals.to_excel(excel2_file_path, index=False)


# Calculate VIFs
vifs = pd.DataFrame()
vifs["Features"] = ["Constant", "Load2", "Load3"]
vifs["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vifs)

# Save VIFs to Excel
vif_file_path = os.path.join(folder_name, 'VIFs.xlsx')
vifs.to_excel(vif_file_path, index=False)

###########
# Table 3 #
###########

summary_betas = summary_table(betas)
print(summary_betas)

excel3_file_path = os.path.join(folder_name, 'Table3_replication.xlsx')
summary_betas.to_excel(excel3_file_path, index=False)


