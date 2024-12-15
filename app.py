### dependencies
## gral
import os
from datetime import datetime
## application
from flask import Flask, jsonify, request, render_template
## pre-process
import pandas as pd
import numpy as np
## model
from lightgbm import LGBMClassifier
import joblib
import json
## explanation
# import lime
# import lime.lime_tabular
# import dill as pickle
## local

## future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

### app instance
app = Flask(__name__, static_folder='images')

### context
with app.app_context():
    ## context parameters
    current_directory = os.path.dirname(__file__)
    # transformations
    ###
    # encodings
    cate_to_oneHotEncoding = []
    # output paths
    output_pred_path = "predictions"
    # feature engine
    ###
    # scalers
    ###
    # models
    model_path = os.path.join(current_directory, "models/model_s217170.pkl")
    model = LGBMClassifier()
    model.load_model(model_path)
    # print(model)
    # model features
    with open(model_path, 'r') as model_file:
        model_info = json.load(model_file)
        model_features = model_info['learner']['feature_names']
    model_file.close()
    # model metadata
    meta_path = os.path.join(current_directory, "models/metadata.json")
    with open(meta_path, 'r') as metadata_file:
        model_meta = json.load(metadata_file)
    metadata_file.close()
    print(model_meta)

### root
@app.route('/')
def index():
    app_name = 'Synth_risk App Server'
    program_name = 'CustomPortal Project'
    author_name = 'SST'
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return render_template('index.html', app_name=app_name, program_name=program_name, 
                           author_name=author_name, current_date=current_date)

### metadata
@app.route('/metadata', methods=['POST'])
def metadata():
    try:
        # getting request
        print('# Model metadata requested #')
        # reading metadata
        with open(meta_path, 'r') as file:
            metadata = json.load(file)
        print(metadata)
        
        return jsonify(metadata)
    
    except Exception as e:
        print(e)
    return jsonify({'error': 'Internal Server Error'}), 500
    
### predict
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    print('Predict batch')
    try:
        # getting request
        X_request = request.get_json(force=True)
        print('------------------------')
        print('------------------------')
        print('# New request #')
        # request as pandas df
        X = pd.DataFrame(X_request)
        print(X)
        # import checking
        X_ = X.copy()
        # feature engine transformation
        X_['not_working'] = np.where(np.in1d(X_['job'], ['student', 'retired', 'unemployed']), 1, 0)
        # one-hot encoding
        X_ = pd.get_dummies(X_, columns=model_meta['cat_to1h'], drop_first=True, dtype=int)
        # column reindexing to fix to model features
        X_ = X_.reindex(columns=model_features, fill_value=0)
        # prediction
        # re-order the features for prediction
        X_ = X_[model_features]
        proba = model.predict_proba(X_)
        # response
        X['proba'] = proba[:, 1]
        print('# Result #')
        print(X[['proba']])
        print('------------------------')
        print('------------------------')
        if proba is not None:
            response_data = X.to_dict()
        else:
            response_data = {'error': 'No prediction generated'}

        return jsonify(response_data)
    
    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)