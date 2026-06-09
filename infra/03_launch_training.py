"""
Launches a SageMaker Training Job for GPT-nano on WikiText-2.

Instance:  ml.g4dn.xlarge  (NVIDIA T4 16 GB, $0.736/hr)
Runtime:   ~40-60 min  →  est. cost ~$0.50-0.75

The job downloads train/val data from S3, runs transformer/train.py,
and saves model.pt + config.json back to S3.

Usage:
  pip install sagemaker boto3
  python infra/03_launch_training.py
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

REGION      = "us-west-2"
BUCKET      = "543458926995-mnist-digit-models"
S3_PREFIX   = "lstm-transformer"
ROLE_NAME   = "lstm-transformer-sagemaker-role"
JOB_NAME    = "lstm-transformer-training"

session  = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
iam      = boto3.client("iam", region_name=REGION)
role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

data_uri = f"s3://{BUCKET}/{S3_PREFIX}/data"
output_uri = f"s3://{BUCKET}/{S3_PREFIX}/output"

estimator = PyTorch(
    entry_point="train.py",
    source_dir="transformer/",           # uploads train.py + model.py + requirements.txt
    role=role_arn,
    framework_version="2.1.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    output_path=output_uri,
    base_job_name=JOB_NAME,
    sagemaker_session=session,
    hyperparameters={
        "vocab-size":  50257,
        "embed-dim":   128,
        "n-heads":     4,
        "n-layers":    4,
        "context-len": 256,
        "ff-dim":      512,
        "dropout":     0.1,
        "batch-size":  32,
        "epochs":      8,
        "lr":          3e-4,
    },
    # Keep logs streaming so you can watch progress
    disable_profiler=True,
)

print(f"Role:   {role_arn}")
print(f"Data:   {data_uri}")
print(f"Output: {output_uri}")
print("\nStarting training job (this window will stream logs) ...")

estimator.fit({"training": data_uri}, wait=True, logs="All")

print("\nTraining complete.")
print(f"Model artifact: {estimator.model_data}")
print("\nNext: update MODEL_DATA_URI in infra/04_deploy_endpoint.py and run it.")
print(f"  MODEL_DATA_URI = \"{estimator.model_data}\"")
