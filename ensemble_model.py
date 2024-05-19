import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from SS3_Model import SS3
from RoBERTa_Model import RoBERTa
from SVM_Model import SVM

# Read the preprocessed data into a DataFrame
# Specify the path to your CSV file
csv_file_path = r'D:\PY\Major Project\Ensemble Model\.venv\preprocessed_data.csv'

df = pd.read_csv(csv_file_path, header=0)

# Drop rows with null values
df = df.dropna()

# Split the DataFrame into features (X) and target labels (y)
X = df['text']
y = df['target']

# Split the dataset into training and testing subsets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X-Train = ")
# print(X_train)
# print("Type of X_train in ensemble model:", type(X_train))

#Training the Base Models
ss3_model = SS3()
ss3_model.train(X_train, y_train)

roberta_model = RoBERTa()
roberta_model.train(X_train, y_train)

svm_model = SVM()
svm_model.train(X_train, y_train)



confusion_matrix = np.zeros((2, 2), dtype=int)
def update_confusion_matrix(conf_matrix, y_true, y_pred):
    if y_true == 4 and y_pred == 4:
        conf_matrix[0][0] += 1  # True Positive
    elif y_true == 0 and y_pred == 4:
        conf_matrix[1][0] += 1  # False Positive
    elif y_true == 4 and y_pred == 0:
        conf_matrix[0][1] += 1  # False Negative
    elif y_true == 0 and y_pred == 0:
        conf_matrix[1][1] += 1  # True Negative

# print("Prediction:")
for text, target in zip(X_test, y_test):
    y_true = target
    y_pred_ss3 = ss3_model.test(text)
    y_pred_roberta = roberta_model.test(text)
    y_pred_svm = svm_model.test(text)

    pos = 0
    neg = 0
    if y_pred_ss3 == 0:
        neg = neg + 1
    else:
        pos = pos + 1
    if y_pred_roberta == 0:
        neg = neg + 1
    else:
        pos = pos + 1
    if y_pred_svm == 0:
        neg = neg + 1
    else:
        pos = pos + 1
    if pos > neg:
        y_pred = 4
    else:
        y_pred = 0

    # print("Given target: ", y_true)
    # print("Predicted target: ", y_pred)
    update_confusion_matrix(confusion_matrix, y_true, y_pred)
    print()

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)


# Calculate precision, recall, F1 score and accuracy from the confusion matrix
def calculate_metrics(conf_matrix):
    tp = conf_matrix[0][0]
    fp = conf_matrix[1][0]
    fn = conf_matrix[0][1]
    tn = conf_matrix[1][1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, f1_score, accuracy

# Using the confusion matrix, calculate the metrics
precision, recall, f1_score, accuracy = calculate_metrics(confusion_matrix)
print()
# Print the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Accuracy: {accuracy:.2f}")