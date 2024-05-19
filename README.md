# Early-Depression-Detection-using-Ensemble-Model-of-SS3-Classifier-RoBERTa-Model-and-SVM

# preprocessing.py
The preprocessing.py file should be executed first to get preprocessed dataset. It is designed to clean and prepare text data for sentiment analysis. It defines a function preprocess_text that processes each text
entry by converting it to lowercase, removing URLs, usernames, mentions, and punctuation, and lemmatizing the text while removing stopwords and rare words. The script reads a CSV file into a DataFrame, applies the
preprocessing function to the 'text' column, and finally saves the cleaned DataFrame to a new CSV file. This preprocessing step is crucial for preparing raw text data for further modeling.

