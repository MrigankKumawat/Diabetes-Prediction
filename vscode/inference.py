import os
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attrbs, cat_attrbs):
    # For Numerical Columns
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown = "ignore")),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrbs),
        ("cat", cat_pipeline, cat_attrbs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("diabetes_prediction_dataset.csv")

    df['age_cat'] = pd.cut(df['age'], bins = [0, 18, 36, 72, np.inf], labels = [1, 2, 3, 4])
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_set, test_set in split.split(df, df['age_cat']):
        strat_train = df.iloc[train_set]
        strat_test = df.iloc[test_set]

    diabetes_train = strat_train.copy()
    diabetes_features_train = diabetes_train.drop(['diabetes', 'age_cat'], axis = 1) 
    diabetes_label_train = diabetes_train['diabetes']
    diabetes_num_train = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    diabetes_cat_train = ['gender', 'smoking_history']

    pipeline = build_pipeline(diabetes_num_train, diabetes_cat_train)
    diabetes_prepared_train = pipeline.fit_transform(diabetes_features_train)
    forest_model = RandomForestClassifier(
        random_state = 42,
        n_estimators = 200,
        max_depth = 15,
        class_weight = 'balanced',
    ) 
    forest_model.fit(diabetes_prepared_train, diabetes_label_train)
    print("Dumping the Model and Pipeline Files.............")
    joblib.dump(forest_model, MODEL_FILE) 
    joblib.dump(pipeline, PIPELINE_FILE) 
    print("Model is Trained and Saved.")
