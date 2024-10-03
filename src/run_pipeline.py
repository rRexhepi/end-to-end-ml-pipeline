mport pandas as pd
import joblib
from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_model(X_train, y_train):
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
    grid_search.fit(X_train, y_train)

    # Return the best estimator
    return grid_search.best_estimator_

def evaluate_model(model, X_val, y_val):
    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate evaluation metrics
    print("Evaluation Metrics:")
    print("-------------------")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def prepare_submission(predictions_df, output_file):
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

    # Load and preprocess the training data
    print("Loading and preprocessing training data...")
    train_df = load_data('data/train.csv')
    y = train_df['Survived']
    X = preprocess_data(train_df, training=True)

    # Split the data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_val, y_val)

    # Save the trained model
    print("Saving the model...")
    save_model(model, 'models/random_forest_model.pkl')

    # Load and preprocess the test data
    print("Loading and preprocessing test data...")
    test_df = load_data('data/test.csv')
    X_test = preprocess_data(test_df, training=False)

    # Make predictions on the test data
    print("Making predictions on the test data...")
    predictions = model.predict(X_test)

    # Save the predictions
    print("Saving predictions...")
    test_df['Predictions'] = predictions
    test_df.to_csv('predictions/predictions.csv', index=False)
    print("Predictions saved to predictions/predictions.csv")

    # Prepare the submission file
    print("Preparing submission file...")
    prepare_submission(test_df, 'predictions/submission.csv')

if __name__ == '__main__':
    main()
