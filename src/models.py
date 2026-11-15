from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import get_preprocessor


def get_models(X_train):
    def make_pipeline(model):
        return Pipeline([
            ("preprocess", get_preprocessor(X_train)),
            ("model", model),
        ])

    models = {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree (depth 5)": make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=42)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = make_pipeline(XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", random_state=42))
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["Gradient Boosting"] = make_pipeline(HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, max_iter=300, random_state=42))

    return models