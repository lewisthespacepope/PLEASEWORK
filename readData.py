import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords')
nltk.download('wordnet')

# Create an instance of the Naive Bayes classifier
classifier = MultinomialNB()

# Path to the dataset directory
dataset_dir = r'C:\Users\USER\PycharmProjects\testAI\aclImdb\train'

# List to store the reviews and labels
reviews = []
labels = []

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

preprocessed_reviews = []

# Iterate through the positive reviews
pos_dir = os.path.join(dataset_dir, 'pos')
for filename in os.listdir(pos_dir):
    file_path = os.path.join(pos_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        review = file.read()
        reviews.append(review)
        labels.append(1)  # Assign label 1 for positive reviews

# Iterate through the negative reviews
neg_dir = os.path.join(dataset_dir, 'neg')
for filename in os.listdir(neg_dir):
    file_path = os.path.join(neg_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        review = file.read()
        reviews.append(review)
        labels.append(0)

for review in reviews:
    # Removing HTML tags and converting to plain text
    soup = BeautifulSoup(review, "html.parser")
    plain_text = soup.get_text()

    # Removing special characters
    review = re.sub('[^\w\s]', '', plain_text)

    # Lowercasing
    review = review.lower()

    # Tokenization
    tokens = review.split()

    # Removing stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Combining tokens into a preprocessed review
    preprocessed_review = " ".join(tokens)
    preprocessed_reviews.append(preprocessed_review)

# Split the preprocessed reviews and labels into training and testing sets
vectorizer = CountVectorizer()
X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, labels, test_size=0.2, random_state=42)

# Learn the vocabulary and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transform the testing data helkkii wirdk
X_test_vectorized = vectorizer.transform(X_test)

X = vectorizer.fit_transform(preprocessed_reviews)

# Convert the sparse matrix to a dense matrix
X = X.toarray()

# Print the shape of the feature matrix
print("Shape of feature matrix:", X.shape)

# Fit the classifier on the training data
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test_vectorized)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Print the first review and its corresponding label
print("Review:", reviews[0])
print("Label:", labels[0])

