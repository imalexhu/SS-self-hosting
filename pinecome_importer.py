import json
import boto3
import logging
import time
from datetime import datetime
import asyncio
import pandas as pd
import requests
from typing import List, Dict
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
import os
import pyarrow as pa
import botocore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pinecone_import_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PineconeImporter:
    def __init__(self, aws_region: str, pinecone_api_key: str, pinecone_index_name: str, pinecone_index_host: str):
        """Initialize PineconeImporter with required credentials and configuration"""
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_index_host = pinecone_index_host
        
        # Initialize AWS client
        session = boto3.Session(region_name=aws_region)
        self.aws_config = botocore.config.Config(
            max_pool_connections=50,
            retries=dict(max_attempts=5)
        )
        self.s3 = session.client(
            service_name='s3',
            region_name=aws_region,
            config=self.aws_config
        )
        
        self.loop = asyncio.get_event_loop()

    async def list_parquet_files(self, bucket: str, prefix: str) -> List[str]:
        """List all parquet files in the specified S3 path"""
        try:
            response = await self.loop.run_in_executor(
                None,
                lambda: self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{bucket}/{prefix}")
                return []
            
            return [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].endswith('.parquet')
            ]
            
        except Exception as e:
            logger.error(f"Error listing parquet files: {str(e)}")
            raise

    async def merge_parquet_files(self, bucket: str, parquet_files: List[str], output_key: str) -> str:
        """Merge multiple parquet files into a single file with formatted metadata."""
        try:
            logger.info(f"Starting merge process for {len(parquet_files)} parquet files...")
            tables = []
            total_rows = 0
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download and read all parquet files
                for i, file_key in enumerate(parquet_files, 1):
                    logger.info(f"Processing file {i}/{len(parquet_files)}: {file_key}")
                    temp_file = os.path.join(temp_dir, os.path.basename(file_key))
                    
                    # Download file
                    download_start = time.time()
                    await self.loop.run_in_executor(
                        None,
                        lambda: self.s3.download_file(bucket, file_key, temp_file)
                    )
                    logger.info(f"Downloaded {file_key} in {time.time() - download_start:.2f}s")
                    
                    # Read table
                    read_start = time.time()
                    table = pq.read_table(temp_file)
                    logger.info(f"Read {table.num_rows} rows from {file_key} in {time.time() - read_start:.2f}s")

                    # Convert to pandas for formatting
                    df = table.to_pandas()

                    # Format metadata as JSON string
                    def combine_metadata(row):
                        metadata_dict = {
                            'abstract': row.get('abstract', None),
                            'doc_index': row.get('doc_index', None),
                            'source': row.get('source', None),
                            'title': row.get('title', None)
                        }
                        return json.dumps(metadata_dict)

                    # Ensure correct column formatting
                    df_formatted = pd.DataFrame({
                        'id': df['id'].astype(str),  # Ensure ID is a string
                        'values': df['values'],      # Vector values column remains unchanged
                        'metadata': df.apply(combine_metadata, axis=1)  # Convert metadata to JSON string
                    })

                    # Convert back to PyArrow Table
                    formatted_table = pa.Table.from_pandas(df_formatted)
                    tables.append(formatted_table)
                    total_rows += formatted_table.num_rows

                    logger.info(f"Formatted table with {formatted_table.num_rows} rows from {file_key}")

                # Concatenate all tables
                logger.info(f"Concatenating {len(tables)} tables with total {total_rows} rows...")
                concat_start = time.time()
                merged_table = pa.concat_tables(tables)
                logger.info(f"Concatenation completed in {time.time() - concat_start:.2f}s")
                
                # Save merged table
                write_start = time.time()
                merged_file = os.path.join(temp_dir, 'merged.parquet')
                
                pq.write_table(merged_table, merged_file)
                logger.info(f"Wrote merged table in {time.time() - write_start:.2f}s")
                
                # Upload merged file to S3
                upload_start = time.time()
                await self.loop.run_in_executor(
                    None,
                    lambda: self.s3.upload_file(merged_file, bucket, output_key)
                )
                logger.info(f"Uploaded merged file to s3://{bucket}/{output_key} in {time.time() - upload_start:.2f}s")
                
                return output_key
                
        except Exception as e:
            logger.error(f"Error merging parquet files: {str(e)}")
            raise

    async def trigger_pinecone_import(self, bucket: str, merged_file_key: str) -> Dict:
        """Trigger Pinecone bulk import using REST API"""
        try:
            storage_uri = f"s3://{bucket}/{merged_file_key}"
            
            import_request = {
                "uri": storage_uri,
                "errorMode": {
                    "onError": "continue"
                }
            }
            
            response = await self.loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"https://{self.pinecone_index_host}/bulk/imports",
                    headers={
                        "Api-Key": self.pinecone_api_key,
                        "Content-Type": "application/json",
                        "X-Pinecone-API-Version": "2025-01"
                    },
                    json=import_request
                )
            )
            
            if response.status_code != 200:
                raise Exception(f"Import failed with status {response.status_code}: {response.text}")
            
            import_response = response.json()
            logger.info(f"Started bulk import from {storage_uri}")
            logger.info(f"Import response: {import_response}")
            
            return import_response
            
        except Exception as e:
            logger.error(f"Error triggering Pinecone import: {str(e)}")
            raise

    async def check_import_status(self, operation_id: str) -> Dict:
        """Check the status of a Pinecone import operation"""
        try:
            response = await self.loop.run_in_executor(
                None,
                lambda: requests.get(
                    f"https://{self.pinecone_index_host}/bulk/imports/{operation_id}",
                    headers={
                        "Api-Key": self.pinecone_api_key,
                        "X-Pinecone-API-Version": "2025-01"
                    }
                )
            )
            
            if response.status_code != 200:
                raise Exception(f"Status check failed with status {response.status_code}: {response.text}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error checking import status: {str(e)}")
            raise

    async def monitor_import_status(self, operation_id: str, check_interval: int = 3):
        """Monitor import status until completion"""
        while True:
            try:
                status = await self.check_import_status(operation_id)
                state = status.get('state', '').lower()
                
                logger.info(f"Import status: {status}")
                
                if state == 'completed':
                    logger.info("Import completed successfully!")
                    break
                elif state in ['failed', 'canceled']:
                    logger.error(f"Import failed or was canceled. Status: {status}")
                    break
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring import status: {str(e)}")
                await asyncio.sleep(check_interval)

    async def check_file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in S3"""
        try:
            await self.loop.run_in_executor(
                None,
                lambda: self.s3.head_object(Bucket=bucket, Key=key)
            )
            return True
        except self.s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    async def process_import(self, bucket: str, base_path: str):
        """Main process to handle the entire import workflow"""
        try:
            merged_file_key = f"{base_path}/merged/final.parquet"
            
            # Check if merged file already exists
            if await self.check_file_exists(bucket, merged_file_key):
                logger.info(f"Found existing merged file at s3://{bucket}/{merged_file_key}")
                logger.info("Proceeding with import using existing merged file...")
            else:
                logger.info("No existing merged file found. Starting merge process...")
                # List all parquet files
                parquet_files = await self.list_parquet_files(bucket, base_path)
                if not parquet_files:
                    raise Exception(f"No parquet files found in s3://{bucket}/{base_path}")
                
                # Merge files
                logger.info(f"Found {len(parquet_files)} parquet files to merge")
                await self.merge_parquet_files(bucket, parquet_files, merged_file_key)
                logger.info("Merge process completed successfully")
            
            # Start import
            logger.info("Starting Pinecone import process...")
            import_response = await self.trigger_pinecone_import(bucket, merged_file_key)
            operation_id = import_response.get('id')
            
            if not operation_id:
                raise Exception("No operation ID received from import response")
            
            logger.info(f"Import started with operation ID: {operation_id}")
            # Monitor import status
            await self.monitor_import_status(operation_id)
            
        except Exception as e:
            logger.error(f"Error in import process: {str(e)}")
            raise
        
async def main():
    """Main function to run the import process"""
    try:
        AWS_REGION = "us-west-2"
        PINECONE_API_KEY = "pcsk_6aWdAr_JzZdKmXzu7MSe8VWbGe5fiz1HZFYgqbSfS67V3T8hQ9va5uiu9U5WzDmmq1hb9H"
        PINECONE_INDEX = "semantic-scholar-aws"
        PINECONE_INDEX_HOST = "semantic-scholar-aws-s13uyyn.svc.apw5-4e34-81fa.pinecone.io"
        BUCKET_NAME = "semantic-s3"
        BASE_PATH = "file1"  # Path where parquet files are stored
        
        
        importer = PineconeImporter(
            aws_region=AWS_REGION,
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX,
            pinecone_index_host=PINECONE_INDEX_HOST
        )
        
        await importer.process_import(BUCKET_NAME, BASE_PATH)
        logger.info("Import process completed")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())