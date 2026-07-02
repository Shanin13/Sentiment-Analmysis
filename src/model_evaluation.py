import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
#from dvclive import Live

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaliating")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_evaluation.log"))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_model(file_path):
    try:
        with open(file_path,"rb") as file:
            model =pickle.load(file)
        logger.debug("Model loaded successfully from {}".format(file_path))
        return model
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from {}".format(file_path))
        return df
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def model_evaluate(x_test,y_test,clf):
    try:
        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics_dict = {"accuracy_score" : accuracy,
                        "Precision_score" : precision,
                        "Recall" : recall

        }
        logger.debug("Metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics, file_path):
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            json.dump(metrics,file)
        logger.debug("Metrics saved successfully to {}".format(file_path))
    except Exception as e:
        logger.error('Error occurred while saving metrics: %s', e)
        raise   

def main():
    try:
        clf = load_model("models/random_forest_model.pkl")
        file = load_data("data/processed/test_tfidf.csv")

        x_test = file.iloc[:,:-1].values
        y_test = file.iloc[:,-1].values
    
        metrics = model_evaluate(x_test,y_test,clf)
        save_metrics(metrics,'reports/metrics.json')

    except Exception as e:
       logger.error('Failed to complete the model evaluation process: %s', e)
       raise

if __name__ == '__main__':
    main()
