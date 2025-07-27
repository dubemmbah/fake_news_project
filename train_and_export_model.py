import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse

# === STEP 1: Load Dataset ===
df_fake = pd.read_csv("./data/Fake.csv")
df_real = pd.read_csv("./data/True.csv")

df_fake["label"] = "FAKE"
df_real["label"] = "REAL"
df = pd.concat([df_fake, df_real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["label_num"] = df["label"].map({"REAL": 0, "FAKE": 1})

# === STEP 2: Text Cleaning Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].astype(str).apply(clean_text)
df["text_length"] = df["clean_text"].apply(len)
df["exclamations"] = df["text"].astype(str).apply(lambda x: x.count("!"))

# === STEP 3: Feature Engineering ===
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_text = tfidf.fit_transform(df["clean_text"])
X_num = df[["text_length", "exclamations"]].values
X = sparse.hstack((X_text, X_num))
y = df["label_num"].values

# === STEP 4: Train Random Forest ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
rf.fit(X_train, y_train)

# === STEP 5: Export Model + Vectorizer ===
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")
