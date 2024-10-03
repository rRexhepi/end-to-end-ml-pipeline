import pandas as pd

def load_data():
    """
    Load the training and test datasets from CSV files.

    Returns:
        train_data (pd.DataFrame): The training dataset with features and target variable.
        test_data (pd.DataFrame): The test dataset with features only.
    """
    # Load the training data
    train_data = pd.read_csv('data/train.csv')

    # Load the test data
    test_data = pd.read_csv('data/test.csv')

    return train_data, test_data

if __name__ == '__main__':
    # This block is optional and can be used for testing the function
    train_data, test_data = load_data()
    print("Training data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)
