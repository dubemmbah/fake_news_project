# app/app.py
from flask import Flask, request, render_template
import pickle
import re
import numpy as np
from newspaper import Article
from bs4 import BeautifulSoup
import requests
from scipy import sparse
import nltk
import spacy
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Initialize summarizer


# summarizer = pipeline(
#     "summarization", 
#     model="facebook/bart-large-cnn", 
#     device=-1,
#     use_safetensors=True
# )

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    use_safetensors=True
)

# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt", device=-1)


app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Dummy domain reputation dictionary
trusted_domains = ["bbc.com", "cnn.com", "reuters.com", "nytimes.com"]
untrusted_domains = ["clickbaitcentral.com", "fakenewsnet.net", "theonion.com"]

# --------- UTIL FUNCTIONS ---------

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.strip()) > 100:
            return article.text.strip()
    except:
        pass

    # Fallback with BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        if len(text.strip()) > 100:
            return text.strip()
    except:
        pass

    return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_domain_reputation(url):
    domain = url.split("//")[-1].split("/")[0].replace("www.", "")
    if domain in trusted_domains:
        return "Trusted", "success"
    elif domain in untrusted_domains:
        return "Untrusted", "danger"
    else:
        return "Unknown", "secondary"

def get_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def summarize_article(text):
    try:
        summary = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return None

# --------- ROUTE ---------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    raw_text = None

    if request.method == "POST":
        url = request.form["url"]
        raw_text = extract_article_text(url)

        if not raw_text:
            return render_template("index.html", result={"error": "Failed to extract article content. Try another link."})

        clean = clean_text(raw_text)
        text_length = len(clean)
        exclamations = raw_text.count("!")
        X_text = vectorizer.transform([clean])
        X_num = np.array([[text_length, exclamations]])
        X_final = sparse.hstack((X_text, X_num))

        pred_proba = model.predict_proba(X_final)[0]
        pred = np.argmax(pred_proba)
        label = "REAL" if pred == 0 else "FAKE"
        confidence = round(np.max(pred_proba) * 100, 2)
        color = "success" if label == "REAL" else "danger"

        domain_reputation, rep_color = get_domain_reputation(url)
        summary = summarize_article(raw_text)
        entities = get_named_entities(raw_text)

        result = {
            "label": label,
            "confidence": f"{confidence}%",
            "color": color,
            "domain_reputation": domain_reputation,
            "rep_color": rep_color,
            "summary": summary,
            "entities": entities
        }

    return render_template("index.html", result=result, raw_text=raw_text)

if __name__ == "__main__":
    app.run(debug=True)
