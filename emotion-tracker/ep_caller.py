import boto3
from sagemaker.predictor import Predictor
import nltk
import pandas as pd
import json
from datetime import datetime

# Setting up AWS session with a specific region
boto3.setup_default_session(region_name="us-east-1")  
nltk.download('punkt')

# Configuring SageMaker predictor
sagemaker_endpoint = "huggingface-pytorch-inference-2024-12-07-17-56-55-417"
predictor = Predictor(endpoint_name=sagemaker_endpoint)

# Configure S3
s3 = boto3.client("s3")
bucket_name = "emotiontrackerbucket"
file_key = "results.csv"

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


    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["Timestamp"] = timestamp

    return results

def update_results_in_s3(new_results):
    """Append new results to the existing CSV file in S3."""
    try:
        s3.download_file(bucket_name, file_key, "results.csv")
        existing_results = pd.read_csv("results.csv")
    except Exception as e:
        print("No existing file found. Creating a new one.")
        existing_results = pd.DataFrame()


    combined_results = pd.concat([existing_results, new_results], ignore_index=True)

    combined_results.to_csv("results.csv", index=False)

    s3.upload_file("results.csv", bucket_name, file_key)

    print("Results updated in S3 successfully.")

# Test input
text = input("Tell me about your day! ")
results = analyze_text(text)
print(results)

# Update results in S3
update_results_in_s3(results)
