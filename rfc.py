import joblib

rfc_model_save_path = "./saved_models/random_forest_model.pkl"

# Load Random Forest model
rfc_model = joblib.load(rfc_model_save_path)

print("RandomForest model loaded successfully.")
