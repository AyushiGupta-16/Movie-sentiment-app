import nltk
import streamlit as st
import random
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Download necessary data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load and preprocess data
documents = [(preprocess_text(movie_reviews.raw(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Add some short real-world reviews
extra_reviews = [
    ("Awesome movie!", "pos"),
    ("Worst film ever", "neg"),
    ("Loved it!", "pos"),
    ("Terrible acting", "neg"),
    ("Brilliant direction", "pos"),
    ("Not worth watching", "neg"),
]
documents.extend([(preprocess_text(text), label) for text, label in extra_reviews])

# Shuffle and split data
random.shuffle(documents)
texts = [doc for doc, label in documents]
labels = [label for doc, label in documents]
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter your review below and click **Predict** to find out if it's Positive or Negative.")

review = st.text_area("Your Review:", height=150)

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess_text(review)
        prediction = model.predict([cleaned])[0]
        sentiment = "🟢 Positive" if prediction == "pos" else "🔴 Negative"
        st.markdown(f"### Predicted Sentiment: {sentiment}")
