from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Load encoders and scalers used during preprocessing
le_embarked = joblib.load('models/le_embarked.pkl')
scaler = joblib.load('models/scaler.pkl')

def preprocess_input(data):
    """
    Preprocess the input data for prediction.
    """
    df = pd.DataFrame([data])

    # Ensure all necessary columns are present
    required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = le_embarked.transform(df['Embarked'])

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1  # Initialize to 1 (Alone)
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # Not alone

    # Scale numerical features
    numerical_features = ['Age', 'Fare', 'FamilySize']
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'SibSp', 'Parch']
    X = df[features]

    return X

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict survival for a single passenger.
    """
    data = request.get_json(force=True)

    # Preprocess the input data
    try:
        X = preprocess_input(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make prediction
    prediction = model.predict(X)
    output = int(prediction[0])
    result = 'Survived' if output == 1 else 'Did not survive'

    return jsonify({'prediction': output, 'result': result})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict survival for multiple passengers.
    """
    data = request.get_json(force=True)

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Preprocess the input data
    try:
        X = preprocess_input_batch(df)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Make predictions
    predictions = model.predict(X)
    outputs = predictions.tolist()
    results = ['Survived' if pred == 1 else 'Did not survive' for pred in outputs]

    return jsonify({'predictions': outputs, 'results': results})

def preprocess_input_batch(df):
    """
    Preprocess batch input data for prediction.
    """
    # Ensure all necessary columns are present
    required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns: {missing_cols}')

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = le_embarked.transform(df['Embarked'])

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1  # Initialize to 1 (Alone)
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # Not alone

    # Scale numerical features
    numerical_features = ['Age', 'Fare', 'FamilySize']
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'SibSp', 'Parch']
    X = df[features]

    return X

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({'status': 'UP'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)