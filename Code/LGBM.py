import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def objective(trial: optuna.Trial):
    df = pd.read_csv("../input/tcc/final_data.csv")
    
    for i in range(5):
        df[f"f{i}"] = pd.cut(df[f"f{i}"], bins=5).codes

    train_x, test_x, train_y, test_y = train_test_split(
        df.drop(columns="target"), df["target"], test_size=0.2
    )

    params = {
        "metric": "auc",
        "objective": "binary",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 1, 100),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(test_x, label=test_y)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    prediction = model.predict(test_x, num_iteration=model.best_iteration)
    return roc_auc_score(test_y, prediction)


study = optuna.create_study()
study.optimize(objective, n_jobs=-1, n_trials=100)
print(study.best_params)
