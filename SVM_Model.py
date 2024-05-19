from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

class SVM:
    def __init__(self):
        # Initialize SVM model parameters
        self.vectorizer = TfidfVectorizer()
        self.svm_model = SVC()

    def train(self, X_train, y_train):
        # Vectorize input texts
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train the SVM model
        self.svm_model.fit(X_train_vec, y_train)

    def test(self, text):
        # Vectorize input text
        text_vec = self.vectorizer.transform([text])

        # Test the SVM model on input text
        predicted_label = self.svm_model.predict(text_vec)[0]

        return predicted_label
