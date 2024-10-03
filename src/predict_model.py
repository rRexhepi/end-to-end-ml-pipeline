import pandas as pd
import joblib
from preprocessing import preprocess_data

def make_predictions(input_data_path, output_path):
    # Load the trained model
    model = joblib.load('models/random_forest_model.pkl')

    # Load new data
    data = pd.read_csv(input_data_path)

    # Preprocess the data
    X = preprocess_data(data)

    # Make predictions
    predictions = model.predict(X)

    # Add predictions to the DataFrame
    data['Predictions'] = predictions

    # Save the predictions
    data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Make predictions with the trained model.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data CSV file.')
    parser.add_argument('--output_data', type=str, required=True, help='Path to save the output CSV file with predictions.')
    args = parser.parse_args()

    make_predictions(args.input_data, args.output_data)
