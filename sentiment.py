import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm  # For progress bar
import os
from dotenv import load_dotenv


# 🔹 Download NLTK data (if not already installed)
nltk.download('vader_lexicon')

load_dotenv()

# ✅ PostgreSQL Connection Using SQLAlchemy (Fixes Password Special Characters)
from urllib.parse import quote
from sqlalchemy import create_engine

# Load environment variables
db_username = os.getenv("DB_USERNAME")
db_password = quote(os.getenv("DB_PASSWORD"))  # Encode special characters
db_name = os.getenv("DB_NAME")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# ✅ Corrected Connection String (Ensures Proper Encoding & Port Definition)
db_url = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# ✅ Create SQLAlchemy Engine
engine = create_engine(db_url)

# ✅ Define Batch Processing Variables
BATCH_SIZE = 50000  # Process 50,000 rows at a time
offset = 0  # Start from the first row
total_processed = 0  # Track processed rows

# 🔹 Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ✅ Optimized Sentiment Analysis (Using `summary` Column)
def batch_sentiment_analysis(texts):
    sentiments = np.full(len(texts), "Neutral", dtype=object)  # Default all to Neutral

    # Handle missing values
    valid_texts = [text if isinstance(text, str) and text.strip() != "" else "No Summary" for text in texts]

    # Compute sentiment scores in batch mode
    vader_scores = [sia.polarity_scores(str(text))['compound'] if text != "No Summary" else 0 for text in valid_texts]
    blob_scores = [TextBlob(str(text)).sentiment.polarity if text != "No Summary" else 0 for text in valid_texts]

    final_scores = np.array(vader_scores) + np.array(blob_scores) / 2

    # 🔹 Adjusted sentiment classification thresholds:
    sentiments[final_scores >= 0.01] = "Positive"  # Was 0.05
    sentiments[final_scores <= -0.01] = "Negative"  # Was -0.05

    return sentiments

# ✅ Process All Rows in Batches
while True:
    print(f"🔹 Fetching {BATCH_SIZE} rows from offset {offset}...")

    # ✅ Fetch a batch of data
    query = f"""
        SELECT id, summary FROM amazon_reviews 
        WHERE summary IS NOT NULL 
        ORDER BY id 
        LIMIT {BATCH_SIZE} OFFSET {offset};
    """
    
    df = pd.read_sql(query, engine)

    # ✅ Stop when no more data is left
    if df.empty:
        print("✅ No more data to process. Exiting loop.")
        break

    print(f"✅ {len(df)} summaries loaded. Processing sentiment analysis...")

    # ✅ Apply Sentiment Analysis
    df["sentiment"] = batch_sentiment_analysis(df["summary"].values)

    # ✅ Prepare data for bulk update
    update_data = [(row["sentiment"], row["id"]) for _, row in df.iterrows()]
    BATCH_UPDATE_SIZE = 10000  # Update in smaller batches

    # ✅ Open psycopg2 Connection for Bulk Update
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()

    # ✅ Split data into smaller batches (PostgreSQL Performance)
    update_batches = [update_data[i:i + BATCH_UPDATE_SIZE] for i in range(0, len(update_data), BATCH_UPDATE_SIZE)]

    # ✅ Process Each Batch Separately (Progress Bar)
    print("🔹 Updating PostgreSQL in batches...")
    for i, batch in enumerate(tqdm(update_batches, desc="Updating DB")):
        execute_values(cursor, """
            UPDATE amazon_reviews
            SET sentiment = data.sentiment
            FROM (VALUES %s) AS data(sentiment, id)
            WHERE amazon_reviews.id = data.id;
        """, batch)
        conn.commit()  # Commit after each batch

    # ✅ Close Connection Properly
    cursor.close()
    conn.close()

    total_processed += len(df)
    offset += BATCH_SIZE  # Move to the next batch

print(f"🎉 Sentiment Analysis completed for {total_processed} rows and data updated in PostgreSQL successfully!")

# ✅ Close SQLAlchemy Engine
engine.dispose()
