import os
import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import joblib

# ✅ Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["SCALER_FOLDER"] = "scalers"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs(app.config["SCALER_FOLDER"], exist_ok=True)

# ✅ Data Cleaning & Preprocessing
def clean_and_preprocess_data(df, target_column, scaler=None, fit_scaler=True):
    df = df.copy()

    # ✅ Clean column names
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True).str.strip("_")

    # ✅ Remove duplicates
    df.drop_duplicates(inplace=True)

    # ✅ Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "MISSING", inplace=True)
        else:
            df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0, inplace=True)

    # ✅ Drop unnecessary columns
    drop_cols = [col for col in df.columns if 'id' in col.lower() or 'timestamp' in col.lower()]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # ✅ Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ✅ Label encode target if classification
    is_classification = len(y.unique()) < 10
    if is_classification and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # ✅ One-Hot Encoding for categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # ✅ Standardize numerical features
    if fit_scaler:
        scaler = StandardScaler()
        X.iloc[:, :] = scaler.fit_transform(X)
    else:
        X.iloc[:, :] = scaler.transform(X)

    return X, y, scaler, is_classification


# ✅ Model Training & Selection
def train_and_compare_models(X, y, is_classification):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Apply SMOTE for classification if imbalanced
    if is_classification and len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # ✅ Define metric
    scoring = 'f1_weighted' if is_classification else 'neg_mean_squared_error'

    # ✅ Model Configurations
    param_grids = {
        "Random Forest": {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        },
        "XGBoost": {
            'model': XGBClassifier(eval_metric="mlogloss" if is_classification else "rmse", random_state=42),
            'params': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}
        },
        "LightGBM": {
            'model': LGBMClassifier(random_state=42),
            'params': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}
        },
        "Logistic Regression": {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.1, 1.0], 'solver': ['liblinear']}
        },
        "KNN": {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}
        }
    }

    best_score = -np.inf
    best_models = {}
    results = {}

    for name, model_info in param_grids.items():
        model = model_info['model']
        param_grid = model_info['params']

        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)

        final_model = search.best_estimator_
        y_pred = final_model.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'accuracy': round(accuracy * 100, 2), 'f1_score': round(f1 * 100, 2)}
            score = f1
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            results[name] = {'rmse': round(rmse, 4)}
            score = -rmse

        if score > best_score:
            best_score = score
            best_models = {name: final_model}  # Reset best models dictionary
        elif score == best_score:
            best_models[name] = final_model  # Store multiple top models

    # ✅ Handle multiple models with the same best score
    if len(best_models) > 1:
        best_model_name = min(best_models, key=lambda x: (len(x), x))  # Choose the simplest model by name length
    else:
        best_model_name = list(best_models.keys())[0]

    best_model = best_models[best_model_name]

    # ✅ Save best model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(app.config["MODEL_FOLDER"], f"{best_model_name}_{timestamp}.pkl")
    joblib.dump(best_model, model_path)

    return results, best_model_name, model_path



# ✅ File Upload & Model Training Route
from flask import Flask, render_template, request, jsonify, redirect, url_for

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(filepath)

            df = pd.read_csv(filepath)
            target_column = request.form.get("target_column")

            if target_column not in df.columns:
                return jsonify({"error": "Invalid target column"}), 400

            X, y, scaler, is_classification = clean_and_preprocess_data(df, target_column)
            results, best_model_name, model_path = train_and_compare_models(X, y, is_classification)

            scaler_path = os.path.join(app.config["SCALER_FOLDER"], f"scaler_{best_model_name}.pkl")
            joblib.dump(scaler, scaler_path)

            return render_template("result.html", 
                                   results=results, 
                                   best_model=best_model_name, 
                                   model_path=model_path, 
                                   scaler_path=scaler_path,
                                   model_type="classification" if is_classification else "regression")

    return render_template("index.html")



# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json.get("data")
    model_path = request.json.get("model_path")
    scaler_path = request.json.get("scaler_path")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    input_df = pd.DataFrame([input_data])

    # ✅ Apply saved scaler
    input_df.iloc[:, :] = scaler.transform(input_df)

    prediction = model.predict(input_df)

    return jsonify({"prediction": prediction.tolist()})


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
