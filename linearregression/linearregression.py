import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("../dataset/healthmate_dataset.csv")

# Encode categorical variables
df["gender"] = df["gender"].map({"male": 0, "female": 1})
df["region"] = df["region"].map({"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3})

# Convert yearly expenses to monthly
df["expenses"] = df["expenses"] / 12  

# Define features (X) and targets (y)
X = df[["age", "bmi", "gender", "region", "children"]]
y = df[["expenses", "premium"]]

# Function to train and evaluate model
def train_and_evaluate(X, y, region, target_name):
    # Filter by region
    region_data = df[df["region"] == region]
    
    X_region = region_data[["age", "bmi", "gender", "region", "children"]]
    y_region = region_data[target_name]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_region, y_region, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

# Evaluate for each region and target
results = {}

region_map = {0: "northeast", 1: "northwest", 2: "southeast", 3: "southwest"}
for region_code, region_name in region_map.items():
    # Expense
    mse_exp, r2_exp = train_and_evaluate(X, y, region_code, "expenses")
    # Premium
    mse_pre, r2_pre = train_and_evaluate(X, y, region_code, "premium")
    
    results[region_name] = {
        "expense": {"MSE": mse_exp, "R2": r2_exp},
        "premium": {"MSE": mse_pre, "R2": r2_pre}
    }

# Print results
for region, metrics in results.items():
    print(f"\nRegion: {region}")
    print(f"  Expense  -> MSE: {metrics['expense']['MSE']:.2f}, R2: {metrics['expense']['R2']:.4f}")
    print(f"  Premium  -> MSE: {metrics['premium']['MSE']:.2f}, R2: {metrics['premium']['R2']:.4f}")
