import os
import logging
import pandas as pd
import ssl
from sklearn.model_selection import train_test_split

# Fix SSL certificate verification error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_injestion")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_injestion.log"))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded successfully from {}".format(file_path))
        return df
    except Exception as e:
        logger.error("Error loading data from {}: {}".format(file_path, e))
        raise

def preprocessing(df):
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error("Error occurred while preprocessing data: {}".format(e))
        raise

def save_data(df, file_path):
    try:
        raw_data_path = os.path.join(file_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=2)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug("Data saved successfully to {}".format(raw_data_path))
    except Exception as e:
        logger.error("Error occurred while saving data: {}".format(e))
        raise
    
def main():
    try:
        data_url = "https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        data = load_data(data_url)
        preprocessed_data = preprocessing(data)
        save_data(preprocessed_data, "./data")

    except Exception as e:
        logger.error("Error occurred in main function: {}".format(e))
        raise

if __name__ == "__main__":
    main()

