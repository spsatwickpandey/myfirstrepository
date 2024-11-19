import joblib
import os

def load_model():
    return joblib.load(os.path.join('model', 'random_forest_model.joblib'))

def load_scaler():
    return joblib.load(os.path.join('model', 'scaler.joblib'))

scaler = load_scaler()