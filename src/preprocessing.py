import os
import logging
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import pandas as pd


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger= logging.getLogger("preprocessing")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "preprocessing.log"))
file_handler.setLevel(logging.DEBUG)    

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)     

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Fix SSL certificate issue and download required NLTK data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)
        
    

def preprocessing(df, text_column, label_column):
    try:
          logger.debug('Starting preprocessing for DataFrame')
          encoder = LabelEncoder()
          df[label_column] = encoder.fit_transform(df[label_column])
          logger.debug('Label encoding completed for column: {}'.format(label_column))
          
          df.drop_duplicates(inplace=True)
          logger.debug('Duplicates removed from DataFrame')

          df.loc[:, text_column] = df[text_column].apply(transform_text)
          logger.debug('Text transformation completed for column: {}'.format(text_column))
          return df
    except Exception as e:
          logger.error('Error occurred during preprocessing: {}'.format(e))
          raise
    
def main():
      try:
            train_data = pd.read_csv('data/raw/train.csv')
            test_data = pd.read_csv('data/raw/test.csv')
            logger.debug('Data loaded successfully for preprocessing')
            train_data_preprocessed = preprocessing(train_data, 'text', 'label')
            test_data_preprocessed = preprocessing(test_data, 'text', 'label')
            
            data_path = os.path.join("./data", "interim")
            os.makedirs(data_path, exist_ok=True)
            
            train_data_preprocessed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
            test_data_preprocessed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
            logger.debug('Preprocessed data saved successfully')    
      
      except Exception as e:
            logger.error('Error occurred in main function: {}'.format(e))
            raise

if __name__ == "__main__":
    main()


