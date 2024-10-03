import pandas as pd
import joblib
from preprocessing import preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("Evaluation Metrics:")
    print("-------------------")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

def main():
    # Load the trained model
    model = joblib.load('models/random_forest_model.pkl')

    # Load and preprocess the validation data
    print("Loading and preprocessing validation data...")
    val_df = load_data('data/validation.csv')  # Ensure you have validation.csv
    y_val = val_df['Survived']
    X_val = preprocess_data(val_df, training=False)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, X_val, y_val)

if __name__ == '__main__':
    main()
