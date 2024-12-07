# To use SageMaker, you need AWS credentials configured. There are several ways to do this:
# 1. Configure AWS CLI with 'aws configure' 
# 2. Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
# 3. Use AWS IAM role if running on AWS services like EC2

import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

# Initialize the SageMaker session with your credentials
session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=session)

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'arpanghoshal/EmoRoBERTa',
	'HF_TASK':'text-classification'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.37.0',
	pytorch_version='2.1.0',
	py_version='py310',
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.m5.xlarge' # ec2 instance type
)

predictor.predict({
	"inputs": "I like you. I love you",
})
