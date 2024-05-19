import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

class RoBERTa:
    def __init__(self):
        # Load pre-trained RoBERTa tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    def train(self, X_train, y_train, batch_size=32, num_epochs=3):
        # Map target labels
        y_train_mapped = y_train.replace({4: 1})

        # Convert input text to tokenized format
        inputs = self.tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(y_train_mapped.tolist())

        # Create a TensorDataset from tokenized inputs and labels
        dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, labels)

        # Use DataLoader for efficient batch loading during training
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(num_epochs):
            for batch_input_ids, batch_attention_mask, batch_labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

    def test(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        # Perform inference
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Predict class based on logits
        predicted_class = np.argmax(logits.detach().numpy())

        # Map predicted class back to original target labels
        return 0 if predicted_class == 0 else 4
