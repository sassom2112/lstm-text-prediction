"""
Creates (or updates) the Lambda function + API Gateway that proxies
to the SageMaker serverless endpoint.

After this script finishes it prints the API Gateway invoke URL.
Update web/package.json predeploy with that URL, then `npm run deploy`.

Usage:
  python infra/05_setup_lambda_api.py
"""
import io
import json
import os
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

REGION        = "us-west-2"
ACCOUNT_ID    = "543458926995"
ENDPOINT_NAME = "lstm-transformer-endpoint"
FUNCTION_NAME = "lstm-transformer-proxy"
API_NAME      = "lstm-transformer-API"
STAGE_NAME    = "prod"

iam     = boto3.client("iam",          region_name=REGION)
lam     = boto3.client("lambda",       region_name=REGION)
apigw   = boto3.client("apigateway",   region_name=REGION)


# ---------------------------------------------------------------------------
# Helper: zip lambda_handler.py in-memory
# ---------------------------------------------------------------------------

def make_zip():
    buf = io.BytesIO()
    handler_path = os.path.join(os.path.dirname(__file__), "lambda_handler.py")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(handler_path, "lambda_handler.py")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Step 1 — Lambda execution role
# ---------------------------------------------------------------------------

def ensure_lambda_role():
    role_name = "lstm-transformer-lambda-role"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": "*",
            },
            {
                "Effect": "Allow",
                "Action": "sagemaker:InvokeEndpoint",
                "Resource": f"arn:aws:sagemaker:{REGION}:{ACCOUNT_ID}:endpoint/{ENDPOINT_NAME}",
            },
        ],
    }
    try:
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"Lambda role exists: {role_arn}")
    except ClientError:
        role_arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
        )["Role"]["Arn"]
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="InvokeSageMaker",
            PolicyDocument=json.dumps(inline_policy),
        )
        print(f"Created Lambda role: {role_arn}")
        print("Waiting 15s for role to propagate ...")
        time.sleep(15)
    return role_arn


# ---------------------------------------------------------------------------
# Step 2 — Lambda function
# ---------------------------------------------------------------------------

def ensure_lambda(role_arn):
    code_zip = make_zip()
    env_vars = {"Variables": {"SAGEMAKER_ENDPOINT_NAME": ENDPOINT_NAME}}
    try:
        fn = lam.get_function(FunctionName=FUNCTION_NAME)
        fn_arn = fn["Configuration"]["FunctionArn"]
        lam.update_function_code(FunctionName=FUNCTION_NAME, ZipFile=code_zip)
        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Environment=env_vars,
        )
        print(f"Updated Lambda: {fn_arn}")
    except ClientError:
        fn_arn = lam.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime="python3.11",
            Role=role_arn,
            Handler="lambda_handler.lambda_handler",
            Code={"ZipFile": code_zip},
            Timeout=60,
            MemorySize=256,
            Environment=env_vars,
        )["FunctionArn"]
        print(f"Created Lambda: {fn_arn}")
    return fn_arn


# ---------------------------------------------------------------------------
# Step 3 — API Gateway REST API
# ---------------------------------------------------------------------------

def ensure_api(fn_arn):
    # Check if API already exists
    apis = apigw.get_rest_apis(limit=100)["items"]
    api = next((a for a in apis if a["name"] == API_NAME), None)

    if api:
        api_id = api["id"]
        print(f"API exists: {api_id}")
    else:
        api_id = apigw.create_rest_api(
            name=API_NAME,
            description="Proxy to lstm-transformer SageMaker endpoint",
        )["id"]
        print(f"Created API: {api_id}")

    # Root resource
    resources = apigw.get_resources(restApiId=api_id)["items"]
    root_id   = next(r["id"] for r in resources if r["path"] == "/")

    # /predict resource
    predict_r = next((r for r in resources if r.get("path") == "/predict"), None)
    if predict_r:
        predict_id = predict_r["id"]
    else:
        predict_id = apigw.create_resource(
            restApiId=api_id, parentId=root_id, pathPart="predict"
        )["id"]
        print("Created /predict resource")

    fn_uri = (
        f"arn:aws:apigateway:{REGION}:lambda:path/2015-03-31/functions/"
        f"{fn_arn}/invocations"
    )

    for method in ["POST", "OPTIONS"]:
        try:
            apigw.get_method(restApiId=api_id, resourceId=predict_id, httpMethod=method)
        except ClientError:
            apigw.put_method(
                restApiId=api_id,
                resourceId=predict_id,
                httpMethod=method,
                authorizationType="NONE",
            )
            apigw.put_integration(
                restApiId=api_id,
                resourceId=predict_id,
                httpMethod=method,
                type="AWS_PROXY",
                integrationHttpMethod="POST",
                uri=fn_uri,
            )
            print(f"  Added {method} → Lambda")

    # Grant API Gateway permission to invoke Lambda
    src_arn = f"arn:aws:execute-api:{REGION}:{ACCOUNT_ID}:{api_id}/*/*/predict"
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="api-gateway-invoke",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=src_arn,
        )
    except ClientError as e:
        if "already exists" not in str(e):
            raise

    # Deploy
    apigw.create_deployment(restApiId=api_id, stageName=STAGE_NAME)
    invoke_url = f"https://{api_id}.execute-api.{REGION}.amazonaws.com/{STAGE_NAME}"
    return invoke_url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Lambda execution role ===")
    role_arn = ensure_lambda_role()

    print("\n=== Step 2: Lambda function ===")
    fn_arn = ensure_lambda(role_arn)

    print("\n=== Step 3: API Gateway ===")
    invoke_url = ensure_api(fn_arn)

    predict_url = f"{invoke_url}/predict"
    print(f"\n{'='*60}")
    print(f"API Gateway URL: {predict_url}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print(f'  1. In web/package.json, update predeploy to use:\n     REACT_APP_API_URL={invoke_url}')
    print("  2. npm run deploy  (from web/)")
    print(f"\n  Or test right now:")
    print(f'  curl -X POST {predict_url} \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"prompt":"the history of science","temperature":0.8,"max_length":20}}\'')


if __name__ == "__main__":
    main()
