import pickle
import os
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join("logs", "model_training.log"))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_date(file_path):
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


def model_training( x_train, y_train,params):
    try:
        if x_train.shape[0] != y_train.shape[0]:
            logger.error("Training data and labels are not aligned.")
            raise ValueError("Training data and labels are not aligned.")
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        clf.fit(x_train, y_train)
        logger.debug("Model training completed successfully.")
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise

def save_model(model, file_path):
    try:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved successfully to {}".format(file_path))
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise    

def main():
    df = pd.read_csv('data/processed/train_tfidf.csv')    
    x_train = df.iloc[:,:-1].values
    y_train = df.iloc[:,-1].values

    clf = model_training(x_train, y_train, params={'n_estimators': 100, 'random_state': 42})

    path = "models/random_forest_model.pkl"
    save_model(clf, path)

    
if __name__ == "__main__":    
    main()
   