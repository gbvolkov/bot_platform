from pathlib import Path
from uuid import uuid4
import mimetypes

import boto3
from botocore.client import Config


from  config import (MINIO_URL
                     , MINIO_BUCKET, 
                     MINIO_ACCESS_KEY, 
                     MINIO_SECRET_KEY)

#ENDPOINT_URL = "https://stage.backend.platform-minio.motorplat.ru"  # use http://... if not TLS
ENDPOINT_URL = MINIO_URL  # use http://... if not TLS
BUCKET = MINIO_BUCKET
ACCESS_KEY = MINIO_ACCESS_KEY
SECRET_KEY = MINIO_SECRET_KEY  # put in env var in real code

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",  # MinIO ignores region but boto3 wants one
)

def upload_and_get_link(file_path: str, prefix: str = "documents/", expires_seconds: int = 3600) -> str:
    p = Path(file_path)
    content_type, _ = mimetypes.guess_type(str(p))
    content_type = content_type or "application/octet-stream"

    key = f"{prefix}{uuid4().hex}_{p.name}"

    s3.upload_file(
        Filename=str(p),
        Bucket=BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET, "Key": key},
        ExpiresIn=expires_seconds,
        HttpMethod="GET",
    )
    return url

if __name__ == "__main__":
    print(upload_and_get_link("./file.pdf"))


