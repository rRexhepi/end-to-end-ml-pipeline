import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df):
    """
    Preprocess the input DataFrame and return the feature matrix X.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        X (pd.DataFrame): The preprocessed feature matrix.
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()

    # Fill missing values
    # Fill missing 'Age' values with the median age
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing 'Embarked' values with the mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fill missing 'Fare' values with the median fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Drop unnecessary columns
    df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True, errors='ignore')

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Load the saved Label Encoder for 'Embarked' or fit a new one
    try:
        le_embarked = joblib.load('models/le_embarked.pkl')
    except FileNotFoundError:
        le_embarked = LabelEncoder()
        le_embarked.fit(df['Embarked'])
        joblib.dump(le_embarked, 'models/le_embarked.pkl')
    df['Embarked'] = le_embarked.transform(df['Embarked'])

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 1  # Initialize to 1 (Alone)
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # Not alone if family size > 1

    # Load the saved scaler or fit a new one
    numerical_features = ['Age', 'Fare', 'FamilySize']
    try:
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        scaler = StandardScaler()
        scaler.fit(df[numerical_features])
        joblib.dump(scaler, 'models/scaler.pkl')
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Select features for modeling
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'SibSp', 'Parch']
    X = df[features].copy()

    return X
