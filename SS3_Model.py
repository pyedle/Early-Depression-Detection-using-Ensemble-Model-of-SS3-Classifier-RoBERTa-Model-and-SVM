import numpy as np
class SS3:
    def __init__(self):
        # Initialize dictionaries to store term-frequency pairs for positive and negative categories
        self.term_freq_positive = {}
        self.term_freq_negative = {}
        self.max_freq_positive = 0
        self.max_freq_negative = 0
        self.total_positive = 0
        self.total_negative = 0

    def train(self, X_train, y_train):
        # Iterate over each document and its corresponding label in the training dataset
        for text, target in zip(X_train, y_train):
            # Split the document into terms (words)
            terms = text.split()
            # Select the appropriate dictionary based on the label
            term_freq_dict = self.term_freq_positive if target == 4 else self.term_freq_negative
            total_words = 0
            # Update term frequencies in the dictionary
            for term in terms:
                total_words += 1
                term_freq_dict[term] = term_freq_dict.get(term, 0) + 1
            if target==4:
                self.total_positive += total_words
            else:
                self.total_negative += total_words

        # Iterate over each key-value pair in the term_freq_positive dictionary
        for term, frequency in self.term_freq_positive.items():
            # Check if the current frequency is greater than the maximum frequency found so far
            if frequency > self.max_freq_positive:
                # If yes, update the maximum frequency
                self.max_freq_positive = frequency

        # Iterate over each key-value pair in the term_freq_positive dictionary
        for term, frequency in self.term_freq_negative.items():
            # Check if the current frequency is greater than the maximum frequency found so far
            if frequency > self.max_freq_negative:
                # If yes, update the maximum frequency
                self.max_freq_negative = frequency

    def test(self, text):
        global_value_positive = 0
        global_value_negative = 0
        terms = text.split()

        for term in terms:
            positive_local_value_term = np.sqrt(self.term_freq_positive.get(term, 0)+1 / self.max_freq_positive+self.total_positive)
            negative_local_value_term = np.sqrt(self.term_freq_negative.get(term, 0)+1 / self.max_freq_negative+self.total_negative)

            median = (positive_local_value_term + negative_local_value_term)/2
            lambda_ = 3 * 1.4826
            val1 = abs(positive_local_value_term-median)
            val2 = abs(negative_local_value_term-median)
            median2 = (val1 + val2)/2

            positive_sigmoid_value_term = 0.5 * np.tanh(4 * (positive_local_value_term - median / (lambda_ * median2)) - 2) + 0.5
            negative_sigmoid_value_term = 0.5 * np.tanh(4 * (negative_local_value_term - median / (lambda_ * median2)) - 2) + 0.5

            positive_sanction_value_term = (1 - positive_sigmoid_value_term)
            negative_sanction_value_term = (1 - negative_sigmoid_value_term)

            global_positive_value_term = positive_local_value_term * positive_sigmoid_value_term * positive_sanction_value_term
            global_value_positive = global_value_positive + global_positive_value_term

            global_negative_value_term = negative_local_value_term * negative_sigmoid_value_term * negative_sanction_value_term
            global_value_negative = global_value_negative + global_negative_value_term

        if(global_value_positive > global_value_negative):
            return 4
        else:
            return 0