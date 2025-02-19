import boto3
import json
import gzip
import io
import logging
import time
from typing import List, Dict, AsyncIterator, Tuple
from datetime import datetime
import asyncio
from tqdm import tqdm
import math
import botocore
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'parquet_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def record_time(self, operation: str, duration: float):
        self.metrics[operation].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'count': len(times)
                }
        return summary
    
    def log_summary(self):
        summary = self.get_summary()
        logger.info("Performance Metrics Summary:")
        for operation, metrics in summary.items():
            logger.info(f"\n{operation}:")
            logger.info(f"  Total time: {metrics['total_time']:.2f}s")
            logger.info(f"  Average time: {metrics['average_time']:.2f}s")
            logger.info(f"  Min time: {metrics['min_time']:.2f}s")
            logger.info(f"  Max time: {metrics['max_time']:.2f}s")
            logger.info(f"  Count: {metrics['count']}")

class ParquetCreator:
    def __init__(self, aws_region: str, max_concurrent_embeddings: int = 50):
        logger.info(f"Initializing ParquetCreator with region: {aws_region}")
        
        self.metrics = MetricsTracker()
        
        # Initialize AWS clients with aggressive concurrency
        session = boto3.Session(region_name=aws_region)
        self.aws_config = botocore.config.Config(
            max_pool_connections=200,
            retries=dict(max_attempts=5),
            read_timeout=30,
            connect_timeout=30,
            tcp_keepalive=True
        )
        
        self.bedrock = session.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            config=self.aws_config
        )
        
        self.s3 = session.client(
            service_name='s3',
            region_name=aws_region,
            config=self.aws_config
        )
        
        self.max_concurrent_embeddings = max_concurrent_embeddings
        self.embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_embeddings)
        self.loop = asyncio.get_event_loop()

    async def stream_gzipped_jsonl(self, bucket: str, key: str, limit: int = None) -> AsyncIterator[Dict]:
        """Stream and parse gzipped JSON Lines file from S3 with optional limit"""
        try:
            start_time = time.time()
            response = await self.loop.run_in_executor(
                None,
                lambda: self.s3.get_object(Bucket=bucket, Key=key)
            )
            s3_time = time.time() - start_time
            self.metrics.record_time('s3_get_object', s3_time)
            logger.info(f"S3 get_object completed in {s3_time:.2f}s")
            
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
                        except Exception as e:
                            logger.warning(f"Error processing line: {e}")
            
            for item in await self.loop.run_in_executor(None, lambda: list(process_gz())):
                yield item
                
        except Exception as e:
            logger.error(f"Error streaming file: {e}")
            raise

    async def get_single_embedding(self, text: str) -> Tuple[List[float], float]:
        async with self.embedding_semaphore:
            start_time = time.time()
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
                duration = time.time() - start_time
                return response_body['embedding'], duration
            except Exception as e:
                logger.error(f"Error getting embedding: {str(e)}")
                raise

    async def get_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[float]]:
        tasks = []
        for text in texts:
            if text.strip():
                tasks.append(self.get_single_embedding(text))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        embeddings = []
        timings = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch embedding: {str(result)}")
                embeddings.append(None)
                timings.append(0)
            else:
                embedding, timing = result
                embeddings.append(embedding)
                timings.append(timing)
        
        return embeddings, timings

    async def process_batch_to_parquet(self, abstracts: List[Dict], embeddings: List[List[float]], 
                                     file_path: str, part_number: int) -> str:
        """Convert batch of vectorized abstracts to Parquet format"""
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
                        'source': file_path,
                        'doc_index': f'doc-{len(data["id"])}'
                    })
            
            # Convert to PyArrow Table
            table = pa.Table.from_pydict(data)
            
            # Create directory structure matching original file path
            base_dir = Path('temp_parquet_files')
            file_dir = base_dir / Path(file_path).parent
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet with original filename structure
            output_path = file_dir / f'parts.{part_number}.parquet'
            pq.write_table(table, str(output_path))
            
            return str(output_path)
        except Exception as e:
            logger.error(f"Error saving to parquet: {str(e)}")
            raise

    async def process_abstracts(self, bucket: str, key: str, batch_size: int = 200, limit: int = None):
        """Process JSON Lines abstracts and store as Parquet files"""
        try:
            logger.info(f"Starting abstract processing with batch size: {batch_size}")
            processed_count = 0
            error_count = 0
            current_texts = []
            current_abstracts = []
            part_number = 0
            
            total_start_time = time.time()
            
            # Extract the file path without .json.gz extension for organizing parquet files
            base_file_path = Path(key).stem.replace('.json', '')

            async for abstract in self.stream_gzipped_jsonl(bucket, key, limit):
                try:
                    text_to_embed = abstract.get('abstract', '')
                    if text_to_embed.strip():
                        current_texts.append(text_to_embed)
                        current_abstracts.append(abstract)
                        
                        if len(current_texts) >= batch_size:
                            # Get embeddings for the batch concurrently
                            batch_start_time = time.time()
                            embeddings, timings = await self.get_embeddings_batch(current_texts)
                            batch_embedding_time = time.time() - batch_start_time
                            
                            # Record timings
                            for timing in timings:
                                self.metrics.record_time('single_embedding', timing)
                            self.metrics.record_time('batch_embedding', batch_embedding_time)
                            
                            # Save batch to Parquet
                            parquet_start = time.time()
                            parquet_path = await self.process_batch_to_parquet(
                                current_abstracts, 
                                embeddings,
                                base_file_path,
                                part_number
                            )
                            
                            # Upload to S3
                            s3_key = f"{base_file_path}/parts.{part_number}.parquet"
                            await self.loop.run_in_executor(
                                None,
                                lambda: self.s3.upload_file(parquet_path, bucket, s3_key)
                            )
                            
                            # Clean up local parquet file
                            os.remove(parquet_path)
                            
                            parquet_time = time.time() - parquet_start
                            self.metrics.record_time('parquet_processing', parquet_time)
                            
                            processed_count += len(embeddings)
                            part_number += 1
                            
                            # Clear batches
                            current_texts = []
                            current_abstracts = []
                            
                            if processed_count % batch_size == 0:
                                self._log_progress(processed_count, error_count, total_start_time)
                
                except Exception as e:
                    logger.error(f"Error processing abstract: {str(e)}")
                    error_count += 1
                    continue

            # Process remaining items if any
            if current_texts:
                embeddings, timings = await self.get_embeddings_batch(current_texts)
                parquet_path = await self.process_batch_to_parquet(
                    current_abstracts,
                    embeddings,
                    base_file_path,
                    part_number
                )
                s3_key = f"{base_file_path}/parts.{part_number}.parquet"
                await self.loop.run_in_executor(
                    None,
                    lambda: self.s3.upload_file(parquet_path, bucket, s3_key)
                )
                os.remove(parquet_path)
            
            total_time = time.time() - total_start_time
            logger.info(f"Processing complete in {total_time:.2f}s")
            logger.info(f"Total processed: {processed_count} with {error_count} errors")
            logger.info(f"Average processing rate: {processed_count/total_time:.2f} records/second")
            self.metrics.log_summary()
            
            return base_file_path  # Return the base path where parquet files were saved
            
        except Exception as e:
            logger.error(f"Fatal error during processing: {str(e)}")
            raise

    def _log_progress(self, processed_count, error_count, total_start_time):
        elapsed_time = time.time() - total_start_time
        rate = processed_count / elapsed_time
        logger.info(f"Progress: {processed_count} records processed ({error_count} errors)")
        logger.info(f"Processing rate: {rate:.2f} records/second")
        self.metrics.log_summary()

async def main():
    """Main function to process articles"""
    logger.info("Starting parquet creation process")
    try:
        AWS_REGION = "us-west-2"
        BUCKET_NAME = "semantic-s3"
        FILE_KEY = "abstracts/file1.json.gz"
        
        creator = ParquetCreator(
            aws_region=AWS_REGION,
            max_concurrent_embeddings=200
        )
        
        base_file_path = await creator.process_abstracts(
            bucket=BUCKET_NAME,
            key=FILE_KEY,
            batch_size=10000,
        )
        
        logger.info(f"Parquet creation completed. Files saved under: {base_file_path}")
        return base_file_path
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())