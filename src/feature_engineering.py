import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)      

file_handler = logging.FileHandler(os.path.join(log_dir, "feature_engineering.log"))
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

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from {}".format(file_path))
        return df
    except Exception as e:
        logger.error("Error loading data from {}: {}".format(file_path, e))
        raise

def apply_tfid(train_data, test_data,max_features):
    try:
        x_train = train_data['text'].fillna('')  # Replace NaN with empty string
        y_train = train_data['label']
        x_test = test_data['text'].fillna('')    # Replace NaN with empty string
        y_test = test_data['label']

        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = tfidf_vectorizer.transform(x_test)
        logger.debug("TF-IDF transformation completed successfully")

        train_df = pd.DataFrame(x_train_tfidf.toarray())
        train_df["label"] = y_train.values

        test_df = pd.DataFrame(x_test_tfidf.toarray())
        test_df["label"] = y_test.values

        logger.debug("TF-IDF transformed dataframes created successfully")
        return train_df, test_df
    except Exception as e:
        logger.error("Error occurred during TF-IDF transformation: {}".format(e))
        raise

def save_data(df, file_path):
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        
        # Save the dataframe to CSV file
        df.to_csv(file_path, index=False)
        logger.debug("Data saved successfully to {}".format(file_path))
    except Exception as e:
        logger.error("Error occurred while saving data: {}".format(e))
        raise
        

def main():
    try:
        params = load_params("/params.yaml")
        max_features = params['feature_engineering']['max_features']
        train_data = load_data('data/interim/train_processed.csv')
        test_data = load_data('data/interim/test_processed.csv')
        train_df, test_df = apply_tfid(train_data, test_data,max_features=max_features)
        save_data(train_df, 'data/processed/train_tfidf.csv')
        save_data(test_df, 'data/processed/test_tfidf.csv')
        logger.debug("Feature engineering completed successfully")
    except Exception as e:
        logger.error("Error occurred in main function: {}".format(e))
        
if __name__ == "__main__":
    main()
          