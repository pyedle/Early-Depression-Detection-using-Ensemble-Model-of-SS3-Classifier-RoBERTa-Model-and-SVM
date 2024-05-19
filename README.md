# Early-Depression-Detection-using-Ensemble-Model-of-SS3-Classifier-RoBERTa-Model-and-SVM

# preprocessing.py
The preprocessing.py file should be executed first to get preprocessed dataset. It is designed to clean and prepare text data for sentiment analysis. It defines a function preprocess_text that processes each text
entry by converting it to lowercase, removing URLs, usernames, mentions, and punctuation, and lemmatizing the text while removing stopwords and rare words. The script reads a CSV file into a DataFrame, applies the
preprocessing function to the 'text' column, and finally saves the cleaned DataFrame to a new CSV file. This preprocessing step is crucial for preparing raw text data for further modeling.

# SS3_Model.py
The SS3_model.py file implements the SS3 sentiment classification model. The SS3 class is defined with an initializer (__init__) that sets up dictionaries to store term frequencies for positive and negative
categories, and tracks the maximum frequency and total word counts for each category. The train method processes the training data by splitting each document into terms, updating term frequencies, and adjusting
total word counts for positive and negative categories. This method also updates the maximum frequency values for the terms in each category. The test method evaluates new text by computing a global sentiment value
based on term frequencies, local, sanction and sigmoid values, ultimately classifying the text as positive (4) or negative (0). This file provides a tailored approach to sentiment analysis by leveraging frequency-
based term evaluation.

# RoBERTa_Model.py
The RoBERTa_Model.py file defines a sentiment classification model using the RoBERTa transformer. The RoBERTa class is defined with an initializer that loads pre-trained RoBERTa tokenizer and model. The train
method fine-tunes the model on the provided training data, converting texts to tokenized format and running several epochs of training using a DataLoader for efficient batch processing. The method also sets up an
optimizer and a loss function for the training process. The test method tokenizes new text, performs inference using the trained model, and predicts the sentiment by analyzing the model's logits output. The
predictions are mapped back to the original sentiment labels (0 or 4). This file leverages the powerful RoBERTa model for sophisticated sentiment analysis.

# SVM_Model.py
The SVM_Model.py file implements a sentiment classification model using a Support Vector Machine (SVM). The SVM class is defined with an initializer that sets up a TF-IDF vectorizer and an SVM model. The train
method vectorizes the input texts and trains the SVM model on the training data. The test method vectorizes new text inputs and predicts their sentiment using the trained SVM model. The predicted label is returned
based on the SVM's classification. This file provides a straightforward machine learning approach to sentiment analysis using the well-known SVM algorithm.

# ensemble_model.py
The ensemble_model.py file combines the SS3, RoBERTa, and SVM models into an ensemble for sentiment analysis. The script reads preprocessed data from a CSV file, drops any rows with null values, and splits the
data into training and testing sets. It then trains each of the three models (SS3, RoBERTa, SVM) on the training data. For the testing phase, it uses each model to predict the sentiment of test data and combines
their predictions using a voting mechanism to decide the final sentiment classification. The ensemble model's performance is evaluated using a confusion matrix, and metrics such as precision, recall, F1 score,
and accuracy are calculated based on the confusion matrix. This file demonstrates how combining multiple models can enhance sentiment analysis performance through an ensemble approach.
