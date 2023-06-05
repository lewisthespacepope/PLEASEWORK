import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sklearn.feature_extraction.text
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()

dataset_dir = r'C:\Users\user\OneDrive\문서\GitHub\PLEASEWORK\easy_ham\easy_ham'
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']


def preprocess_email(email):
    soup = BeautifulSoup(email, "html.parser")
    plain_text = soup.get_text()

    # Removing special characters
    email = re.sub('[^\w\s]', '', plain_text)

    # Lowercasing
    email = email.lower()

    # Tokenization
    tokens = email.split()

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Joining tokens back to text
    preprocessed_email = ' '.join(stemmed_tokens)

    return preprocessed_email



# Initialize the list of emails and labels
emails = []
labels = []

for filename in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, filename)
    if os.path.isfile(file_path):
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    email_content = file.read()
                    emails.append(email_content)
                    # Append a label based on the email category (spam/ham)
                    if "spam" in filename:
                        labels.append(1)  # Spam email
                    else:
                        labels.append(0)  # Ham (non-spam) email
                break  # Break the loop if decoding succeeds
            except UnicodeDecodeError:
                continue  # Try the next encoding

        # Handle cases where none of the encodings were successful
        else:
            print(f"Could not decode file: {file_path}")

# Preprocess each email
preprocessed_emails = []
for email in emails:
    preprocessed_email = preprocess_email(email)
    preprocessed_emails.append(preprocessed_email)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the preprocessed emails
features = vectorizer.fit_transform(preprocessed_emails)

# Convert the features to a dense representation
features = features.toarray()

# Get the feature names (vocabulary) learned by the vectorizer
vocabulary = vectorizer.get_feature_names()

# Print the shape of the features
print("Features shape:", features.shape)
print("Vocabulary:", vocabulary)



# Split the features and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)



# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)