import pandas as pd
import re
import string
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs, usernames, mentions, and hashtags using regex
    text = re.sub(r'(http\S+)|(@[^\s]+)|(#[^\s]+)|(@\w+)', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Parse the text using spaCy
    doc = nlp(text)
    # Lemmatize each token, remove stopwords, and remove rare words
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.text.isalpha() and len(token.text) > 1])
    return lemmatized_text

# Read the CSV file into a DataFrame
df = pd.read_csv(r'D:\PY\Major Project\training.1600000.processed.noemoticon.csv', encoding='latin1', usecols=[0, 5], names=['target', 'text'])

# Preprocess the 'text' column
df['text'] = df['text'].apply(preprocess_text)

print(df)

# Save the preprocessed DataFrame to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)
