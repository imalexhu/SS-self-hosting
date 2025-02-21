import asyncio
import sys
import boto3
import logging
from datetime import datetime
from typing import List, Dict, AsyncIterator, Optional, Tuple
import json
import gzip
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import os
import time
import botocore
from concurrent.futures import ThreadPoolExecutor
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'wave_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Print AWS environment variables
print("AWS Environment Variables:")
print(f"AWS_ACCESS_KEY_ID present: {'AWS_ACCESS_KEY_ID' in os.environ}")
print(f"AWS_SECRET_ACCESS_KEY present: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
print(f"AWS_REGION present: {'AWS_REGION' in os.environ}")

class ParquetCreator:
    def __init__(self, aws_region: str, max_concurrent_embeddings: int = 50):
        self.aws_config = botocore.config.Config(
            max_pool_connections=500,
            retries=dict(max_attempts=5),
            read_timeout=30,
            connect_timeout=30,
            tcp_keepalive=True
        )
        
        session = boto3.Session(region_name=aws_region)
        self.bedrock = session.client('bedrock-runtime', config=self.aws_config)
        self.s3 = session.client('s3', config=self.aws_config)
        
        self.embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_embeddings)
        self.loop = asyncio.get_event_loop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.thread_pool.shutdown(wait=True)

    async def stream_gzipped_jsonl(self, bucket: str, key: str, limit: int = None) -> AsyncIterator[Dict]:
        try:
            response = await self.loop.run_in_executor(
                None,
                lambda: self.s3.get_object(Bucket=bucket, Key=key)
            )
            
            count = 0
            def process_gz():
                nonlocal count
                with gzip.GzipFile(fileobj=response['Body']) as gz:
                    for line in gz:
                        if limit and count >= limit:
                            break
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str:
                                count += 1
                                yield json.loads(line_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON line: {e}")

            for item in await self.loop.run_in_executor(None, lambda: list(process_gz())):
                yield item
                
        except Exception as e:
            logger.error(f"Error streaming file: {e}")
            raise

    async def get_single_embedding(self, text: str) -> List[float]:
        async with self.embedding_semaphore:
            try:
                request_body = {
                    "inputText": text,
                    "dimensions": 512,
                    "normalize": True
                }
                
                response = await self.loop.run_in_executor(
                    self.thread_pool,
                    lambda: self.bedrock.invoke_model(
                        modelId="amazon.titan-embed-text-v2:0",
                        contentType="application/json",
                        accept="*/*",
                        body=json.dumps(request_body)
                    )
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['embedding']
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}")
                raise

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        tasks = []
        for text in texts:
            if text.strip():
                tasks.append(self.get_single_embedding(text))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def save_to_parquet(self, abstracts: List[Dict], embeddings: List[List[float]], 
                            base_path: str, part_number: int) -> str:
        try:
            data = {
                'id': [],
                'values': [],
                'metadata': []
            }
            
            for abstract, embedding in zip(abstracts, embeddings):
                if embedding is not None:
                    data['id'].append(f'doc-{len(data["id"])}')
                    data['values'].append(embedding)
                    data['metadata'].append({
                        'title': abstract.get('corpusid', ''),
                        'abstract': abstract.get('abstract', ''),
                        'source': base_path,
                        'doc_index': f'doc-{len(data["id"])}'
                    })
            
            # Convert to PyArrow Table
            table = pa.Table.from_pydict(data)
            
            # Create directory structure
            base_dir = Path('temp_parquet_files')
            file_dir = base_dir / Path(base_path).parent
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet
            output_path = file_dir / f'parts.{part_number}.parquet'
            pq.write_table(table, str(output_path))
            
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            raise

    async def process_batch(self, abstracts: List[Dict], bucket: str, base_path: str, part_number: int):
        try:
            texts = [a.get('abstract', '').strip() for a in abstracts]
            texts = [t for t in texts if t]  # Filter empty texts
            
            embeddings = await self.get_embeddings_batch(texts)
            
            if embeddings:
                parquet_path = await self.save_to_parquet(
                    abstracts, embeddings, base_path, part_number)
                
                s3_key = f"{base_path}/parts.{part_number}.parquet"
                await self.loop.run_in_executor(
                    None,
                    lambda: self.s3.upload_file(parquet_path, bucket, s3_key)
                )
                os.remove(parquet_path)
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    async def process_abstracts(self, bucket: str, key: str, batch_size: int = 1000):
        base_file_path = Path(key).stem.replace('.json', '')
        processed_count = 0
        current_batch = []
        part_number = 0
        
        try:
            async for abstract in self.stream_gzipped_jsonl(bucket, key):
                current_batch.append(abstract)
                
                if len(current_batch) >= batch_size:
                    await self.process_batch(current_batch, bucket, base_file_path, part_number)
                    processed_count += len(current_batch)
                    part_number += 1
                    logger.info(f"Processed {processed_count} abstracts")
                    current_batch = []
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
            
            # Process remaining batch
            if current_batch:
                await self.process_batch(current_batch, bucket, base_file_path, part_number)
                processed_count += len(current_batch)
            
            return base_file_path
            
        except Exception as e:
            logger.error(f"Error processing abstracts: {str(e)}")
            raise

class WaveProcessor:
    def __init__(self, 
                 aws_region: str,
                 source_bucket: str,
                 files_prefix: str = "abstracts/",
                 wave_size: int = 4,
                 embeddings_per_instance: int = 40):
        
        self.aws_region = aws_region
        self.source_bucket = source_bucket
        self.files_prefix = files_prefix
        self.wave_size = wave_size
        self.embeddings_per_instance = embeddings_per_instance
        
        # Initialize S3 client with proper configuration
        self.aws_config = botocore.config.Config(
            max_pool_connections=50,
            retries=dict(max_attempts=5)
        )
        self.s3 = boto3.client('s3', region_name=aws_region, config=self.aws_config)
        
        # Create processing queue
        self.queue = asyncio.Queue()
        
    async def list_input_files(self) -> List[str]:
        try:
            files = []
            paginator = self.s3.get_paginator('list_objects_v2')
            
            async for page in self.paginate(paginator, Bucket=self.source_bucket, Prefix=self.files_prefix):
                if 'Contents' in page:
                    files.extend(obj['Key'] for obj in page['Contents'] 
                               if obj['Key'].endswith('.json.gz'))
            
            if not files:
                logger.warning(f"No files found in {self.source_bucket}/{self.files_prefix}")
            else:
                logger.info(f"Found {len(files)} files to process")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise

    async def paginate(self, paginator, **kwargs):
        for page in paginator.paginate(**kwargs):
            yield page
            await asyncio.sleep(0)

    def chunk_files_into_waves(self, files: List[str]) -> List[List[str]]:
        if not files:
            return []
            
        waves = [files[i:i + self.wave_size] 
                for i in range(0, len(files), self.wave_size)]
        
        logger.info(f"Created {len(waves)} waves of up to {self.wave_size} files each")
        return waves

    async def process_single_file(self, file_key: str) -> None:
        try:
            logger.info(f"Starting processing of file: {file_key}")
            
            async with ParquetCreator(
                aws_region=self.aws_region,
                max_concurrent_embeddings=self.embeddings_per_instance
            ) as creator:
                await creator.process_abstracts(
                    bucket=self.source_bucket,
                    key=file_key,
                    batch_size=1000
                )
                
            logger.info(f"Completed processing file: {file_key}")
            
        except Exception as e:
            logger.error(f"Error processing {file_key}: {str(e)}")
            raise

    async def process_wave_with_queue(self, wave_files: List[str], wave_num: int) -> None:
        logger.info(f"Starting wave {wave_num} with {len(wave_files)} files")
        
        # Initialize queue with files
        for file_key in wave_files:
            await self.queue.put(file_key)
        
        # Create workers
        workers = []
        for _ in range(min(len(wave_files), self.wave_size)):
            worker = asyncio.create_task(self.queue_worker(wave_num))
            workers.append(worker)
        
        # Wait for queue to empty
        await self.queue.join()
        
        # Cancel workers
        for worker in workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        logger.info(f"Completed wave {wave_num}")

    async def queue_worker(self, wave_num: int):
        while True:
            try:
                file_key = await self.queue.get()
                try:
                    await self.process_single_file(file_key)
                except Exception as e:
                    logger.error(f"Error in wave {wave_num} processing {file_key}: {str(e)}")
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                break

    async def process_all_waves(self) -> None:
        try:
            all_files = await self.list_input_files()
            waves = self.chunk_files_into_waves(all_files)
            
            logger.info(f"Starting processing of {len(all_files)} files in {len(waves)} waves")
            
            for wave_num, wave_files in enumerate(waves, 1):
                await self.process_wave_with_queue(wave_files, wave_num)
                
                if wave_num < len(waves):
                    logger.info(f"Waiting between waves {wave_num} and {wave_num + 1}")
                    await asyncio.sleep(3)
            
            logger.info("All waves completed successfully")
            
        except Exception as e:
            logger.error(f"Error in wave processing: {str(e)}\n{traceback.format_exc()}")
            raise

async def main():
    try:
        logger.info("Starting wave processing")
        
        processor = WaveProcessor(
            aws_region="us-west-2",
            source_bucket="semantic-s3",
            wave_size=5,
            embeddings_per_instance=100
        )
        
        await processor.process_all_waves()
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        logger.info("Process complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error in main thread: {str(e)}")
        sys.exit(1)
        