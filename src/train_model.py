import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from data_loader import load_data
from preprocessing import preprocess_data
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import argparse

def train_model(model_name='logistic_regression'):
    """
    Train a machine learning model and log experiments with MLflow.

    Args:
        model_name (str): The type of model to train ('logistic_regression' or 'random_forest').
    """
    # Load the data
    train_data, _ = load_data()

    # Separate the target variable
    y = train_data['Survived']

    # Preprocess the data
    X = preprocess_data(train_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start an MLflow run
    with mlflow.start_run():
        # Define the model based on the input argument
        if model_name == 'logistic_regression':
            # Baseline Model: Logistic Regression
            model = LogisticRegression(solver='liblinear', random_state=42)
        elif model_name == 'random_forest':
            # Advanced Model: Random Forest with Hyperparameter Tuning
            model = RandomForestClassifier(random_state=42)

            # Define hyperparameters for Grid Search
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }

            # Perform Grid Search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            # Fit the Grid Search to find the best hyperparameters
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            # Log best hyperparameters
            mlflow.log_params(grid_search.best_params_)
        else:
            print(f"Model '{model_name}' is not supported.")
            return

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_pred = model.predict(X_val)

        # Evaluate the model
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        print(f"Model: {model_name}")
        print(f"Validation Accuracy: {acc:.4f}")
        print("\nClassification Report:\n", report)
        print("Confusion Matrix:\n", cm)

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)

        # Log the model
        mlflow.sklearn.log_model(model, f"{model_name}-model")

        # Save the model locally
        joblib.dump(model, f"models/{model_name}_model.pkl")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        help='Type of model to train: logistic_regression or random_forest.')
    args = parser.parse_args()

    train_model(model_name=args.model)