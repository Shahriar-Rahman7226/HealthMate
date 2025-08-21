import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv("../dataset/healthmate_dataset.csv")

# Convert yearly expenses to monthly
df['expenses'] = df['expenses'] / 12  

# Map categorical variables
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['region'] = df['region'].map({'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3})

# Features and targets
features = ['age', 'gender', 'bmi', 'children']  
targets = ['expenses', 'premium']

# MLP with ReLU
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()

        self.fc4 = nn.Linear(32, output_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        x = self.act3(self.bn3(self.fc3(x)))
        return self.fc4(x)

results = {}

# Iterate region-wise
for region_code, region_name in zip(range(4), ['northeast', 'northwest', 'southeast', 'southwest']):
    df_region = df[df['region'] == region_code]

    X = df_region[features].values
    y = df_region[targets].values

    # Skip if not enough data
    if len(df_region) < 30:
        print(f"Skipping {region_name}, not enough samples")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features and targets separately
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model
    model = MLP(input_dim=X_train.shape[1], output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # Training with early stopping
    best_loss = np.inf
    patience, patience_counter = 50, 0

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch} for region {region_name}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).numpy()

    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    # Metrics
    mse_exp = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    r2_exp = r2_score(y_true[:, 0], y_pred[:, 0])
    mse_prem = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    r2_prem = r2_score(y_true[:, 1], y_pred[:, 1])

    results[region_name] = {
        "mse_expense": mse_exp, "r2_expense": r2_exp,
        "mse_premium": mse_prem, "r2_premium": r2_prem
    }

# Print results
for region, metrics in results.items():
    print(f"Region: {region}")
    print(f"  Monthly Expense - MSE: {metrics['mse_expense']:.2f}, R2: {metrics['r2_expense']:.2f}")
    print(f"  Premium - MSE: {metrics['mse_premium']:.2f}, R2: {metrics['r2_premium']:.2f}")
    print()
