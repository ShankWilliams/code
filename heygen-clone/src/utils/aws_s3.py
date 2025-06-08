import boto3
from flask import current_app

def get_s3_client(config=None):
    """Create an AWS S3 client using provided config or Flask app config."""
    if config is None:
        config = current_app.config if current_app else {}
    return boto3.client(
        's3',
        aws_access_key_id=config.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config.get('AWS_SECRET_ACCESS_KEY'),
        region_name=config.get('AWS_REGION', 'us-east-1'),
    )
