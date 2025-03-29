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

# ✅ Initialize Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["SCALER_FOLDER"] = "scalers"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs(app.config["SCALER_FOLDER"], exist_ok=True)


# ✅ Home Route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ✅ Upload & Process Route
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        target_column = request.form.get("target_column", "").strip()

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # ✅ Load dataset
        df = pd.read_csv(file_path)

        # ✅ Preprocess Data
        X, y, scaler, is_classification = clean_and_preprocess_data(df, target_column)

        # ✅ Train Models
        results, best_model_name, model_path = train_and_compare_models(X, y, is_classification)

        # ✅ Suggest Hyperparameters
        hyperparameters = suggest_hyperparameters(X, y)

        # ✅ Send Data to `result.html`
        return render_template(
            "result.html",
            results=results,
            best_model=best_model_name,
            model_path=model_path,
            scaler_path="N/A",  # If you saved the scaler, update this
            model_type="classification" if is_classification else "regression",
            hyperparameters=hyperparameters  # ✅ Avoids JSON serialization error
        )

    except Exception as e:
        return jsonify({"error": f"Error processing dataset: {str(e)}"}), 500


# ✅ Data Preprocessing
def clean_and_preprocess_data(df, target_column):
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
    scaler = StandardScaler()
    X.iloc[:, :] = scaler.fit_transform(X)

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
    best_model = None
    best_model_name = ""

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
            best_model = final_model
            best_model_name = name

    # ✅ Save Best Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(app.config["MODEL_FOLDER"], f"{best_model_name}_{timestamp}.pkl")
    joblib.dump(best_model, model_path)

    return results, best_model_name, model_path


# ✅ Suggest Hyperparameters
def suggest_hyperparameters(X, y):
    return {
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "XGBoost": {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "Logistic Regression": {"C": [0.1, 1.0], "solver": ["liblinear"]},
        "KNN": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
    }


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
