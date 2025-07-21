import boto3
from botocore.exceptions import ClientError
import os
import sys
from typing import List
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv()

class S3Manager:
    def __init__(self):
        self.s3_boto_client = None
        self.bucket_name = os.getenv("AWS_BUCKET_NAME", "partgenie-data")
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the S3 client with credentials from environment variables"""
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        aws_region = os.getenv("AWS_REGION")

        if not all([aws_access_key, aws_secret_key, aws_region]):
            print("Error: Missing AWS credentials in environment variables", file=sys.stderr)
            return

        try:
            self.s3_boto_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                config=boto3.session.Config(signature_version='s3v4')
            )
        except Exception as e:
            print(f"Error initializing S3 client: {e}", file=sys.stderr)

    def list_pdfs_in_category(self, series_name: str, category_id: str) -> List[str]:
        """List PDF files in the specified category/series folder"""
        if not self.s3_boto_client:
            print("S3 client not initialized", file=sys.stderr)
            return []

        prefix = f"series_category_pdf_links/{category_id}/{series_name}/"
        print(f"\nSearching in: s3://{self.bucket_name}/{prefix}")
        
        pdf_keys = []
        
        try:
            paginator = self.s3_boto_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".pdf") and not key.endswith('/'):
                            pdf_keys.append(key)
                            print(f"Found PDF: {key}")
            
            print(f"Total PDFs found: {len(pdf_keys)}")
            return pdf_keys
            
        except ClientError as e:
            print(f"S3 Error: {e.response['Error']['Message']}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def generate_presigned_urls(self, keys: List[str], expiration: int = 1800) -> List[str]:
        """Generate presigned URLs for a list of S3 keys that open in browser"""
        if not self.s3_boto_client:
            print("S3 client not initialized", file=sys.stderr)
            return []

        urls = []
        for key in keys:
            try:
                url = self.s3_boto_client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': self.bucket_name,
                        'Key': key,
                        'ResponseContentDisposition': 'inline',
                        'ResponseContentType': 'application/pdf'  # Explicit content type
                    },
                    ExpiresIn=expiration
                )
                urls.append(url)
                print(f"Generated URL for: {key}")
            except ClientError as e:
                print(f"Error generating URL for {key}: {e.response['Error']['Message']}")
            except Exception as e:
                print(f"Unexpected error with {key}: {e}")
        
        return urls