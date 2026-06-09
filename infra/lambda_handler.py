"""
Lambda function — thin CORS proxy between API Gateway and the SageMaker endpoint.
boto3 is included in the Lambda Python runtime, no extra packages needed.
Env vars (set automatically by 05_setup_lambda_api.py):
  SAGEMAKER_ENDPOINT_NAME  — name of the serverless endpoint
"""
import json
import os

import boto3

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
_runtime = boto3.client("sagemaker-runtime")

CORS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
    "Content-Type":                 "application/json",
}


def lambda_handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
        resp = _runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(body).encode(),
        )
        result = json.loads(resp["Body"].read())
        return {"statusCode": 200, "headers": CORS, "body": json.dumps(result)}
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": CORS,
            "body": json.dumps({"error": str(exc)}),
        }
