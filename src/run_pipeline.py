import pandas as pd
import joblib
from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os

def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def train_model(X, y):
    """
    Train a RandomForestClassifier model using GridSearchCV.
    """
    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # Fit the model
    grid_search.fit(X, y)

    # Return the best estimator
    return grid_search.best_estimator_

def save_model(model, model_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def prepare_submission(predictions_df, output_file):
    """
    Prepare the submission file in the required format.
    """
    submission = pd.DataFrame({
        'PassengerId': predictions_df['PassengerId'],
        'Survived': predictions_df['Predictions']
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

def main():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)

    # Step 1: Load and preprocess the training data
    print("Loading and preprocessing training data...")
    train_df = load_data('data/train.csv')
    y = train_df['Survived']
    X = preprocess_data(train_df)

    # Step 2: Train the model
    print("Training the model...")
    model = train_model(X, y)

    # Step 3: Save the trained model
    print("Saving the model...")
    save_model(model, 'models/random_forest_model.pkl')

    # Step 4: Load and preprocess the test data
    print("Loading and preprocessing test data...")
    test_df = load_data('data/test.csv')
    X_test = preprocess_data(test_df)

    # Step 5: Make predictions on the test data
    print("Making predictions on the test data...")
    predictions = model.predict(X_test)

    # Step 6: Save the predictions
    print("Saving predictions...")
    test_df['Predictions'] = predictions
    test_df.to_csv('predictions/predictions.csv', index=False)
    print("Predictions saved to predictions/predictions.csv")

    # Step 7: Prepare the submission file (optional)
    print("Preparing submission file...")
    prepare_submission(test_df, 'predictions/submission.csv')

if __name__ == '__main__':
    main()