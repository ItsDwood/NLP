import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from util import get_dataloader

train_data = get_dataloader("train", 4, True)
test_data = get_dataloader("dev", 4, True)

# Vectorize the text data using normalized word counts
vectorizer = CountVectorizer()
train_input = vectorizer.fit_transform(train_data["Sentence"])
test_input = vectorizer.transform(test_data["Sentence"])

# Train the Naive Bayes classifier
Naive_Bayes = MultinomialNB()
Naive_Bayes.fit(train_input, train_data["Sentiment"])

# Predict sentiment on the test set
predictions = Naive_Bayes.predict(test_input)

# Calculate confusion matrix
confusionMatrix = confusion_matrix(test_data["Sentiment"], predictions)

# Extracting true positives, false positives, true negatives, and false negatives
true_positives = np.diag(confusionMatrix)
false_positives = np.sum(confusionMatrix, axis=0) - true_positives
false_negatives = np.sum(confusionMatrix, axis=1) - true_positives
true_negatives = np.sum(confusionMatrix) - (true_positives + false_positives + false_negatives)

# Print the results
for i, label in enumerate(["neutral", "positive", "negative"]):
    print(f"Class: {label}")
    print(f"True Positives: {true_positives[i]}")
    print(f"False Positives: {false_positives[i]}")
    print(f"True Negatives: {true_negatives[i]}")
    print(f"False Negatives: {false_negatives[i]}\n")

# Classification report
print("\nClassification Report:")
print(classification_report(test_data["Sentiment"], predictions))