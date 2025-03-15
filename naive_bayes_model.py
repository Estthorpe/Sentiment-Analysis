# ✅ Essential Libraries
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote

# ✅ Scikit-Learn Libraries for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ✅ PostgreSQL Connection (Fix Special Characters in Password)
username = "postgres"
password = "Xdv22{TM"
encoded_password = quote(password)  # URL-encode special characters
database_name = "amazon_reviews"

# ✅ Connection String
db_url = f"postgresql://{username}:{encoded_password}@localhost:5432/{database_name}"

# ✅ Create SQLAlchemy Engine
engine = create_engine(db_url)

# ✅ Load Data from PostgreSQL
print("🔹 Fetching data from PostgreSQL...")
query = "SELECT summary, sentiment FROM amazon_reviews WHERE sentiment IS NOT NULL;"
df = pd.read_sql(query, engine)
engine.dispose()  # Close DB connection

# ✅ Balance the Dataset (Undersample Positive Class to Match Negative & Neutral)
negative_samples = df[df["sentiment"] == "Negative"]
neutral_samples = df[df["sentiment"] == "Neutral"]
positive_samples = df[df["sentiment"] == "Positive"].sample(len(negative_samples))  # Undersample Positive

df_balanced = pd.concat([negative_samples, neutral_samples, positive_samples])  # Create balanced dataset
print(f"✅ Balanced Dataset: {df_balanced['sentiment'].value_counts()}")

# ✅ Basic Text Preprocessing (Lowercasing Only)
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""  # Handle missing values safely

    return text.lower()  # ✅ Convert to lowercase without tokenization

# ✅ Apply Preprocessing
print("🔹 Preprocessing text data...")
df_balanced["cleaned_summary"] = df_balanced["summary"].apply(preprocess_text)

# ✅ Remove Empty Rows After Preprocessing
df_balanced = df_balanced[df_balanced["cleaned_summary"].str.strip() != ""]

# ✅ Encode Sentiment Labels
print("🔹 Encoding sentiment labels...")
df_balanced["sentiment_label"] = df_balanced["sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})

# ✅ Split Data (Train: 80%, Test: 20%)
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["cleaned_summary"], df_balanced["sentiment_label"], test_size=0.2, random_state=42
)

# ✅ Define Optimized Naïve Bayes Pipeline
print("🔹 Training Naïve Bayes Model...")
model = make_pipeline(
    CountVectorizer(ngram_range=(1, 3), stop_words="english", max_features=10000),  # Uses 3-grams, limit features
    TfidfTransformer(sublinear_tf=True),  # Smoother term frequency scaling
    MultinomialNB(alpha=0.1)  # Reduce smoothing effect
)

# ✅ Train Model
model.fit(X_train, y_train)

# ✅ Evaluate Model
print("🔹 Evaluating Model Performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# ✅ Display Classification Report
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))


import joblib

# ✅ Save the model
joblib.dump(model, "naive_bayes_sentiment_model.pkl")
print("✅ Model saved successfully as 'naive_bayes_sentiment_model.pkl'")

# ✅ Load the model
model = joblib.load("naive_bayes_sentiment_model.pkl")

# ✅ Function to Predict Sentiment
def predict_sentiment(text):
    text = text.lower()
    prediction = model.predict([text])[0]
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment_map[prediction]

# ✅ Example Predictions
print(predict_sentiment("I love this product! It's amazing."))
print(predict_sentiment("It's okay, nothing special."))
print(predict_sentiment("This is the worst experience ever!"))
