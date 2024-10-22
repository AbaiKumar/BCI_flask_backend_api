from flask import Flask, request
from flask_cors import CORS
from collections import Counter
from joblib import load
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome"

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        # Save the uploaded file
        file.save('test_data_from_app.csv')

        # Example new data (replace this with your new data for prediction)
        new_data = pd.read_csv("test_data_from_app.csv")

        new_data.dropna(inplace= True)
        new_data.drop_duplicates(inplace= True)
        new_data = new_data[~(new_data == 0.0).any(axis=1)]
        new_data = new_data.iloc[10:-10]
        
        model_filename = f'./random_forest_classifier_generalized_compressed.joblib'
        loaded_model = load(model_filename)

        for col in new_data.select_dtypes(include=['float64']).columns:
            new_data[col] = new_data[col].map(lambda x: f'{x:.6f}')

        predictions = loaded_model.predict(new_data)
        class_counts = Counter(predictions)
        max_key = max(class_counts, key=class_counts.get)
        return max_key

    
    except Exception as e:
        return str(e)

app.run(host="0.0.0.0", debug=True)