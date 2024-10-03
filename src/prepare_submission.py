import pandas as pd
import argparse

def prepare_submission(predictions_file, output_file):
    """
    Prepare the submission file in the required format.

    Args:
        predictions_file (str): Path to the CSV file containing predictions.
        output_file (str): Path to save the formatted submission file.
    """
    # Load the predictions
    predictions_df = pd.read_csv(predictions_file)

    # Ensure that 'PassengerId' and 'Predictions' columns exist
    if 'PassengerId' not in predictions_df.columns or 'Predictions' not in predictions_df.columns:
        raise ValueError("The predictions file must contain 'PassengerId' and 'Predictions' columns.")

    # Prepare the submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': predictions_df['PassengerId'],
        'Survived': predictions_df['Predictions']
    })

    # Save the submission file
    submission.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare the submission file.')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to the CSV file containing predictions.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the formatted submission file.')
    args = parser.parse_args()

    prepare_submission(args.predictions_file, args.output_file)