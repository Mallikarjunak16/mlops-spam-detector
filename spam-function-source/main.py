import pandas as pd
import joblib
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import scipy.sparse as sp
from google.cloud import storage

# --- Configuration ---
BUCKET_NAME = 'spam-detector-assets' # Your bucket name

# --- Feature Engineering Functions (Copied from your notebook) ---
def contains_link(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return 1 if url_pattern.search(text) else 0

def contains_upi_keywords(text):
    upi_keywords = ['upi', 'kyc', 'otp', 'credited', 'debited', 'account blocked', 'payment request', 'pin']
    text_lower = text.lower()
    return 1 if any(keyword in text_lower for keyword in upi_keywords) else 0

def uses_url_shortener(text):
    shortener_pattern = re.compile(r'bit\.ly|tinyurl\.com|goo\.gl')
    return 1 if shortener_pattern.search(text) else 0

def special_char_count(text):
    special_chars = ['!', '$', '%', '*', '@', '#', '&']
    return sum(text.count(char) for char in special_chars)

def is_trusted_link(text):
    TRUSTED_DOMAINS = ['sbi.co.in', 'hdfcbank.com', 'icicibank.com', 'axisbank.com', 'rbi.org.in', 'google.com', 'amazon.in']
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    match = url_pattern.search(text)
    if not match: return 0
    url = match.group(0)
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        if domain in TRUSTED_DOMAINS: return 1
    except: return 0
    return 0

# --- Main Retraining Function ---
def retrain_model(request):
    """
    This function is designed to be triggered by a cloud event.
    It loads data, retrains the model, and saves it back to GCS.
    """
    print("Starting model retraining...")
    
    # 1. Load data and model from GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    # Load dataset from GCS
    blob = bucket.blob('spam.csv')
    blob.download_to_filename('/tmp/spam.csv')

    # === THIS IS THE CORRECTED CODE BLOCK ===
    print("Reading CSV data...")
    df = pd.read_csv(
        '/tmp/spam.csv',
        encoding='latin1',
        usecols=['v1', 'v2']
    )
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    # =======================================
    
    # 2. Perform Feature Engineering
    print("Performing feature engineering...")
    df['contains_link'] = df['message'].apply(contains_link)
    df['has_upi_keyword'] = df['message'].apply(contains_upi_keywords)
    df['uses_shortener'] = df['message'].apply(uses_url_shortener)
    df['special_char_count'] = df['message'].apply(special_char_count)
    df['is_trusted_link'] = df['message'].apply(is_trusted_link)
    
    # 3. Prepare data for XGBoost
    print("Preparing data for training...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_text = vectorizer.fit_transform(df['message'])
    X_numerical = df[['contains_link', 'has_upi_keyword', 'uses_shortener', 'special_char_count', 'is_trusted_link']].values
    X_combined = sp.hstack([X_text, X_numerical], format='csr')
    y = df['label']
    
    # 4. Retrain the model
    print("Retraining XGBoost model...")
    model = XGBClassifier(
        learning_rate=0.2,
        max_depth=3,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_combined, y)
    
    # 5. Save the newly trained model and vectorizer back to GCS
    print("Saving updated model and vectorizer to GCS...")
    joblib.dump(model, '/tmp/best_model.pkl')
    joblib.dump(vectorizer, '/tmp/vectorizer.pkl')
    
    bucket.blob('best_model.pkl').upload_from_filename('/tmp/best_model.pkl')
    bucket.blob('vectorizer.pkl').upload_from_filename('/tmp/vectorizer.pkl')
    
    print("Retraining complete!")
    return "OK"