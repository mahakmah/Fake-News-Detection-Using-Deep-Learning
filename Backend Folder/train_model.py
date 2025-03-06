import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset (Ensure dataset contains both real and fake news)
df_fake = pd.read_csv("fake_news/Fake.csv")
df_real = pd.read_csv("fake_news/True.csv")

# Add labels (1 for fake, 0 for real)
df_fake["label"] = 1
df_real["label"] = 0

# Balance dataset (ensure equal fake and real samples)
min_size = min(len(df_fake), len(df_real))
df_fake = df_fake.sample(n=min_size, random_state=42)
df_real = df_real.sample(n=min_size, random_state=42)

df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)  # Shuffle

# Select relevant columns
df = df[['text', 'label']]

# Remove null values
df.dropna(inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create and train the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
