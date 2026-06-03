import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load FAQs
questions = []
answers = []

with open("faqs.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if "|" in line:
            q, a = line.split("|", 1)
            questions.append(q)
            answers.append(a)

# Text preprocessing
def preprocess(text):

    text = text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

# Process FAQ questions
processed_questions = [
    preprocess(q)
    for q in questions
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)

# Chat function
def get_response(user_question):

    processed_input = preprocess(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_input]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarities.argmax()

    confidence = similarities[0][best_match_index]

    if confidence < 0.2:
        return (
            "Sorry, I couldn't find a matching answer."
        )

    return answers[best_match_index]

# Chat Loop
print("=" * 50)
print("FAQ CHATBOT")
print("Type 'exit' to quit")
print("=" * 50)

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)

    print("Bot:", response)