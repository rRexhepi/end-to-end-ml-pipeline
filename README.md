# End-to-End Machine Learning Pipeline

This project demonstrates a complete machine learning workflow using Python, focusing on the Titanic dataset to predict passenger survival.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- `pip` package installer

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/end-to-end-ml-pipeline.git
   cd end-to-end-ml-pipeline

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt

3. **Download the Dataset**
    - Place train.csv and test.csv in the data/ directory.
    - Obtain the dataset from the Kaggle Titanic Competition.

### Usage

1. **Run the pipeline**
Execute the entire machine learning pipeline:
    ```bash

    python src/run_pipeline.py

This script will:
    - Load and preprocess data
    - Train the model
    - Save the model and preprocessing objects
    - Make predictions 
    - Save predictions and prepare a submission file

2. **Start the API**
Launch the Flask API to serve predictions:
    ```bash

    python src/app.py

3. **Test the API**
Using test_api.py
    ```bash

    python src/test_api.py

Using curl
Single Prediction:
    ```bash

    curl -X POST -H "Content-Type: application/json" \
     -d '{
           "Pclass": 3,
           "Sex": "male",
           "Age": 22,
           "SibSp": 1,
           "Parch": 0,
           "Fare": 7.25,
           "Embarked": "S"
         }' \
     http://localhost:5000/predict

### Project Structure
    - data/: Datasets
    - models/: Trained model and preprocessing objects
    - predictions/: Prediction outputs and submission files
    - src/: Source code scripts
    - requirements.txt: Python dependencies
    - README.md: Project documentation

### Dependencies
Install the required packages:
    ```bash

    pip install -r requirements.txt

Key Packages:
    - pandas
    - numpy
    - scikit-learn
    - Flask
    - requests
    - joblib
    