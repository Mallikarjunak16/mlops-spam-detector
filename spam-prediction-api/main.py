import joblib
import re
from urllib.parse import urlparse
import scipy.sparse as sp
from google.cloud import storage
import functions_framework

# --- Configuration ---
BUCKET_NAME = 'spam-detector-assets' # Your bucket name

# --- Global Variables ---
model = None
vectorizer = None

# --- Feature Engineering Functions ---
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

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")

# Load model and vectorizer at startup
print("Loading model...")
download_from_gcs(BUCKET_NAME, 'best_model.pkl', '/tmp/best_model.pkl')
model = joblib.load('/tmp/best_model.pkl')

print("Loading vectorizer...")
download_from_gcs(BUCKET_NAME, 'vectorizer.pkl', '/tmp/vectorizer.pkl')
vectorizer = joblib.load('/tmp/vectorizer.pkl')
print("Model and vectorizer loaded successfully.")

@functions_framework.http
def predict_spam(request):
    """HTTP Cloud Function to predict if a message is spam."""
    # === THIS IS THE NEW CODE BLOCK TO ADD CORS HEADERS ===
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    # =======================================================

    if not model or not vectorizer:
        return ("Error: Model or vectorizer not loaded.", 500, headers)

    request_json = request.get_json(silent=True)
    if not request_json or 'message' not in request_json:
        return ("Error: JSON payload must include a 'message' key.", 400, headers)

    message = request_json['message']
    print(f"Received message for prediction: {message}")

    # Create features
    X_text = vectorizer.transform([message])
    numerical_features = [
        contains_link(message),
        contains_upi_keywords(message),
        uses_url_shortener(message),
        special_char_count(message),
        is_trusted_link(message)
    ]
    X_numerical = sp.csr_matrix([numerical_features])
    X_combined = sp.hstack([X_text, X_numerical], format='csr')
    
    # Make prediction
    prediction_proba = model.predict_proba(X_combined)[0][1]
    prediction = "spam" if prediction_proba > 0.5 else "ham"
    
    print(f"Prediction: {prediction} (Probability: {prediction_proba:.4f})")
    
    response_data = {
        "prediction": prediction,
        "spam_probability": float(prediction_proba)
    }
    
    # Return the response data with the CORS headers
    return (response_data, 200, headers)