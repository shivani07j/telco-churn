from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def get_preprocessor(X):
    categorical_cols = [c for c in X.columns if c not in NUMERIC_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical_cols),
        ],
        sparse_threshold=0,
    )