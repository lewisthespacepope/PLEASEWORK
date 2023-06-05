import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Download the NLTK sentence tokenizer

dataSetDir = r"C:\Users\USER\PycharmProjects\testAI\work producitivity etc.docx.txt"


def preprocess_text(text):
    # Preprocessing steps (lowercasing, removing punctuation, tokenization, etc.)
    # Add your specific preprocessing steps based on your requirements
    # For example:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.sent_tokenize(text)
    return tokens


with open(dataSetDir, 'r', encoding='utf-8') as file:
    file_content = file.read()

preprocessed_text = preprocess_text(file_content)


def search_theme(sentences, theme):
    relevant_sentences = []
    for sentence in sentences:
        if re.search(fr'\b{theme}\b', sentence, flags=re.IGNORECASE):
            relevant_sentences.append(sentence)
    return relevant_sentences


# Example usage
theme = 'Leadership'
relevant_sentences = search_theme(preprocessed_text, theme)

# Print the relevant sentences
for sentence in relevant_sentences:
    print(sentence)
