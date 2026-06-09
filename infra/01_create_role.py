"""
Creates the IAM role that SageMaker will assume for training + inference.
Run once. Safe to re-run — skips creation if role already exists.

Usage:
  python infra/01_create_role.py
"""
import json
import boto3
from botocore.exceptions import ClientError

ROLE_NAME = "lstm-transformer-sagemaker-role"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

MANAGED_POLICIES = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
]

iam = boto3.client("iam")


def main():
    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        print(f"Role already exists: {role['Arn']}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="SageMaker execution role for lstm-transformer project",
        )["Role"]
        print(f"Created role: {role['Arn']}")

        for policy_arn in MANAGED_POLICIES:
            iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
            print(f"  Attached {policy_arn.split('/')[-1]}")

    print(f"\nRole ARN (paste into 03_launch_training.py if needed):\n  {role['Arn']}")


if __name__ == "__main__":
    main()
