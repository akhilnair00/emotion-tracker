import boto3
from sagemaker.predictor import Predictor
import nltk
import pandas as pd
import json
import time

# setting up AWS session with a specific region
boto3.setup_default_session(region_name="us-east-1")  # Replace with your region
nltk.download('punkt')

# configuring SageMaker predictor
sagemaker_endpoint = "huggingface-pytorch-inference-2024-12-06-19-12-20-756"
predictor = Predictor(endpoint_name=sagemaker_endpoint)

def analyze_text(text):
    """Tokenize input text and analyze emotions using SageMaker."""
    sentences = nltk.sent_tokenize(text)

    serialized_input = json.dumps({"inputs": sentences})

    response = predictor.predict(
        serialized_input,
        initial_args={"ContentType": "application/json"}
    )

    decoded_response = json.loads(response.decode("utf-8"))

    results = pd.DataFrame([
        {"Sentence": sentence, "Emotion": result["label"], "Score": result["score"]}
        for sentence, result in zip(sentences, decoded_response)
    ])
    return results

# test input
text = input("Tell me about your day!")
results = analyze_text(text)
print(results)

# import to s3 bucket
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"results_{timestamp}.csv"
results.to_csv("journalentry.csv", index=False)
s3 = boto3.client("s3")
s3.upload_file("filename", "emotiontrackerbucket", "filename")