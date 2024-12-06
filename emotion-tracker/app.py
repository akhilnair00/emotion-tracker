import os
import boto3
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk

# Flask App
app = Flask(__name__)

# Configure S3
s3 = boto3.client("s3")
BUCKET_NAME = "emotionmodelakhil"
MODEL_PATH = "/tmp/emoroberta_model"

# Ensure NLTK tokenizer is available
nltk.download('punkt')

def download_model():
    """Download model files from S3 to /tmp directory."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH, exist_ok=True)
        # List of model files in S3
        model_files = [
            "config.json", "merges.txt", "model.safetensors",
            "special_tokens_map.json", "tokenizer_config.json",
            "tokenizer.json", "vocab.json"
        ]
        for file in model_files:
            s3.download_file(BUCKET_NAME, f"emoroberta_model/{file}", os.path.join(MODEL_PATH, file))

def load_emotion_model():
    """Load the EmoRoBERTa model from /tmp directory."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize the model pipeline
download_model()
emotion_pipeline = load_emotion_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze emotions in a given journal entry."""
    data = request.json
    paragraph = data.get("journal_text", "")

    if not paragraph:
        return jsonify({"error": "No journal text provided."}), 400

    # Tokenize paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)

    # Analyze emotions for each sentence
    results = []
    for sentence in sentences:
        result = emotion_pipeline(sentence)[0]
        results.append({
            'Sentence': sentence,
            'Emotion': result['label'],
            'Score': result['score']
        })

    # Return results as JSON
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
