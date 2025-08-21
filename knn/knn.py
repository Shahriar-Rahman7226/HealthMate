import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
df = pd.read_csv("../dataset/healthmate_dataset.csv")

# Map gender and region to integers
df["gender"] = df["gender"].map({"male": 0, "female": 1})
region_mapping = {region: i for i, region in enumerate(sorted(df["region"].unique()))}
df["region"] = df["region"].map(region_mapping)

# Features
numeric_features = ["age", "bmi", "children"]
categorical_features = ["gender", "region"]
features = numeric_features + categorical_features

X = df[features]
y_premium = df["premium"]
y_expenses_annual = df["expenses"]

# Split
X_train, X_test, y_prem_train, y_prem_test, y_exp_train_yr, y_exp_test_yr = train_test_split(
    X, y_premium, y_expenses_annual, test_size=0.2, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Set a suitable number of neighbors
optimal_neighbors = 7

# Train models
prem_model = KNeighborsRegressor(n_neighbors=optimal_neighbors, weights="distance", metric="minkowski")
exp_model = KNeighborsRegressor(n_neighbors=optimal_neighbors, weights="distance", metric="manhattan")

prem_model.fit(X_train_scaled, y_prem_train)
exp_model.fit(X_train_scaled, y_exp_train_yr)

# Predict
y_prem_pred = prem_model.predict(X_test_scaled)
y_exp_pred_yr = exp_model.predict(X_test_scaled)

# Convert yearly expenses to monthly
y_exp_pred_mo = y_exp_pred_yr / 12.0
y_exp_test_mo = y_exp_test_yr.values / 12.0

# Prepare results per region
results = []
region_map = {v: k for k, v in region_mapping.items()}  # reverse mapping

for region in sorted(df["region"].unique()):
    mask = X_test_scaled["region"] == region
    if mask.sum() == 0:
        continue  # skip if no test samples for this region

    mse_prem = mean_squared_error(y_prem_test[mask], y_prem_pred[mask])
    r2_prem = r2_score(y_prem_test[mask], y_prem_pred[mask])

    mse_exp = mean_squared_error(y_exp_test_mo[mask], y_exp_pred_mo[mask])
    r2_exp = r2_score(y_exp_test_mo[mask], y_exp_pred_mo[mask])

    results.append({
        "region": region_map[region],
        "premium(mse)": mse_prem,
        "premium(r2)": r2_prem,
        "expense(mse)": mse_exp,
        "expense(r2)": r2_exp
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Format for readability
pd.set_option('display.float_format', '{:,.2f}'.format)
print(results_df)
