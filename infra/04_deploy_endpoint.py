"""
Deploys the trained GPT-nano as a SageMaker Serverless Inference endpoint.

Serverless config:
  Memory: 3072 MB  (model ~25 MB, PyTorch runtime ~1.5 GB)
  Max concurrency: 3
  Cold start: ~15-30s (first request after idle)
  Cost: $0 when idle, ~$0.20 per million inference-units

Before running:
  1. Training job must have completed (script 03)
  2. Set MODEL_DATA_URI to the model.tar.gz S3 path printed by script 03
     OR let this script look it up automatically from the latest training job.

Usage:
  python infra/04_deploy_endpoint.py
"""
import json
import os
import shutil
import subprocess
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

REGION         = "us-west-2"
BUCKET         = "543458926995-mnist-digit-models"
S3_PREFIX      = "lstm-transformer"
ROLE_NAME      = "lstm-transformer-sagemaker-role"
ENDPOINT_NAME  = "lstm-transformer-endpoint"

# ── Set this to the model artifact URI printed by script 03 ──────────────────
# Leave as None to auto-discover from the most recent training job.
MODEL_DATA_URI = "s3://543458926995-mnist-digit-models/lstm-transformer/local_output/model.tar.gz"
# ─────────────────────────────────────────────────────────────────────────────

session   = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
sm_client = boto3.client("sagemaker", region_name=REGION)
iam       = boto3.client("iam",       region_name=REGION)
s3        = boto3.client("s3",        region_name=REGION)
role_arn  = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]


def auto_discover_model_uri():
    """Find model.tar.gz from the most recent lstm-transformer training job."""
    jobs = sm_client.list_training_jobs(
        NameContains="lstm-transformer",
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["TrainingJobSummaries"]
    if not jobs:
        raise RuntimeError("No completed lstm-transformer training job found.")
    job_name = jobs[0]["TrainingJobName"]
    detail   = sm_client.describe_training_job(TrainingJobName=job_name)
    uri      = detail["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Auto-discovered model artifact from job '{job_name}':\n  {uri}")
    return uri


def repack_model_with_inference_code(original_uri: str) -> str:
    """
    SageMaker training jobs save model.pt + config.json in model.tar.gz.
    We need to add code/inference.py, code/model.py, code/requirements.txt
    so the PyTorch serving container can find the custom handler.
    Returns the S3 URI of the repacked archive.
    """
    import boto3.s3.transfer

    with tempfile.TemporaryDirectory() as tmpdir:
        original_tar = os.path.join(tmpdir, "original.tar.gz")

        # Download original archive
        bucket, key = original_uri.replace("s3://", "").split("/", 1)
        print(f"Downloading {original_uri} ...")
        s3.download_file(bucket, key, original_tar)

        # Extract
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)
        with tarfile.open(original_tar) as tf:
            tf.extractall(extract_dir)

        # Copy inference code
        code_dst = os.path.join(extract_dir, "code")
        os.makedirs(code_dst, exist_ok=True)
        code_src = os.path.join(os.path.dirname(__file__), "..", "transformer", "code")
        model_src = os.path.join(os.path.dirname(__file__), "..", "transformer", "model.py")
        for fname in ["inference.py", "requirements.txt"]:
            shutil.copy(os.path.join(code_src, fname), os.path.join(code_dst, fname))
        shutil.copy(model_src, os.path.join(code_dst, "model.py"))

        # Repack
        repacked_tar = os.path.join(tmpdir, "model-repacked.tar.gz")
        with tarfile.open(repacked_tar, "w:gz") as tf:
            for item in os.listdir(extract_dir):
                tf.add(os.path.join(extract_dir, item), arcname=item)
        print("Repacked with inference code.")

        # Upload
        repack_key = f"{S3_PREFIX}/repacked/model.tar.gz"
        s3.upload_file(repacked_tar, BUCKET, repack_key)
        repacked_uri = f"s3://{BUCKET}/{repack_key}"
        print(f"Uploaded repacked model to {repacked_uri}")
        return repacked_uri


def main():
    model_uri = MODEL_DATA_URI or auto_discover_model_uri()
    repacked_uri = repack_model_with_inference_code(model_uri)

    model = PyTorchModel(
        model_data=repacked_uri,
        role=role_arn,
        framework_version="2.1.0",
        py_version="py310",
        sagemaker_session=session,
        name="lstm-transformer-model",
    )

    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=3072,
        max_concurrency=3,
    )

    print(f"Deploying to serverless endpoint '{ENDPOINT_NAME}' ...")
    print("(This takes ~5 min the first time)")
    predictor = model.deploy(
        serverless_inference_config=serverless_config,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"\nEndpoint active: {ENDPOINT_NAME}")

    # Quick smoke test
    import json
    resp = predictor.predict(
        {"prompt": "the history of science", "temperature": 0.8, "max_length": 15}
    )
    print(f"Smoke test response:\n{json.dumps(resp, indent=2)}")
    print("\nNext: python infra/05_setup_lambda_api.py")


if __name__ == "__main__":
    main()
