import boto3
import json
import gzip
import io
import logging
from typing import List, Dict, AsyncIterator
from datetime import datetime
import asyncio
from tqdm import tqdm
import math
from pinecone import Pinecone
import botocore
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'document_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, aws_region: str, pinecone_api_key: str, pinecone_index_name: str,
                 max_concurrent_embeddings: int = 5):
        logger.info(f"Initializing DocumentProcessor with region: {aws_region}")
        
        # Initialize AWS clients
        session = boto3.Session(region_name=aws_region)
        self.aws_config = botocore.config.Config(
            max_pool_connections=50,
            retries=dict(max_attempts=10)
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
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(pinecone_index_name)
        
        self.max_concurrent_embeddings = max_concurrent_embeddings
        self.embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_embeddings)
        self.loop = asyncio.get_event_loop()

    async def stream_gzipped_jsonl(self, bucket: str, key: str) -> AsyncIterator[Dict]:
        """Stream and parse gzipped JSON Lines file from S3"""
        try:
            response = await self.loop.run_in_executor(
                None,
                lambda: self.s3.get_object(Bucket=bucket, Key=key)
            )
            
            def process_gz():
                with gzip.GzipFile(fileobj=response['Body']) as gz:
                    for line in gz:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str:
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

    async def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
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
        """Get embeddings for multiple texts concurrently"""
        async with self.embedding_semaphore:
            # Create tasks for all texts in the batch
            tasks = [self.get_single_embedding(text) for text in texts]
            
            # Execute all embedding tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, replacing exceptions with None
            embeddings = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch embedding: {str(result)}")
                    embeddings.append(None)
                else:
                    embeddings.append(result)
            
            return embeddings

    async def process_abstracts(self, bucket: str, key: str, batch_size: int = 100):
        """Process JSON Lines abstracts and store embeddings in Pinecone"""
        try:
            logger.info(f"Starting abstract processing with batch size: {batch_size}")
            processed_count = 0
            error_count = 0
            current_batch = []
            current_texts = []
            current_abstracts = []

            # Stream and process abstracts
            async for abstract in self.stream_gzipped_jsonl(bucket, key):
                try:
                    text_to_embed = abstract.get('abstract', '')
                    if text_to_embed.strip():
                        current_texts.append(text_to_embed)
                        current_abstracts.append(abstract)
                        
                        # Process batch when it reaches the specified size
                        if len(current_texts) >= batch_size:
                            # Get embeddings for the batch concurrently
                            embeddings = await self.get_embeddings_batch(current_texts)
                            
                            # Create vectors for valid embeddings
                            for abstract, embedding in zip(current_abstracts, embeddings):
                                if embedding is not None:
                                    current_batch.append({
                                        'id': f'doc-{processed_count}',
                                        'values': embedding,
                                        'metadata': {
                                            'title': abstract.get('corpusid', ''),
                                            'abstract': abstract.get('abstract', ''),
                                            'source': f"{bucket}/{key}",
                                            'doc_index': f'doc-{processed_count}',
                                        }
                                    })
                                    processed_count += 1
                                else:
                                    error_count += 1
                            
                            # Upsert the batch to Pinecone
                            if current_batch:
                                await self.loop.run_in_executor(
                                    None,
                                    lambda: self.index.upsert(vectors=current_batch)
                                )
                                logger.info(f"Successfully upserted batch of {len(current_batch)} vectors. Total processed: {processed_count}")
                            
                            # Clear batches
                            current_batch = []
                            current_texts = []
                            current_abstracts = []
                            
                            if processed_count % 1000 == 0:
                                logger.info(f"Progress: {processed_count} records processed ({error_count} errors)")
                
                except Exception as e:
                    logger.error(f"Error processing abstract: {str(e)}")
                    error_count += 1
                    continue

            # Process any remaining items
            if current_texts:
                embeddings = await self.get_embeddings_batch(current_texts)
                for abstract, embedding in zip(current_abstracts, embeddings):
                    if embedding is not None:
                        doc_id = abstract.get('id', f'doc-{processed_count}')
                        current_batch.append({
                            'id': str(doc_id),
                            'values': embedding,
                            'metadata': {
                                'title': abstract.get('title', ''),
                                'abstract': abstract.get('abstract', ''),
                                'source': f"{bucket}/{key}",
                                'doc_index': doc_id,
                            }
                        })
                        processed_count += 1
                    else:
                        error_count += 1
                
                if current_batch:
                    await self.loop.run_in_executor(
                        None,
                        lambda: self.index.upsert(vectors=current_batch)
                    )
                    logger.info(f"Successfully upserted final batch of {len(current_batch)} vectors")
            
            logger.info(f"Processing complete. Total processed: {processed_count} with {error_count} errors")
            
        except Exception as e:
            logger.error(f"Fatal error during processing: {str(e)}")
            raise

async def main():
    logger.info("Starting document processing")
    try:
        AWS_REGION = "us-west-2"  
        PINECONE_API_KEY = "pcsk_6aWdAr_JzZdKmXzu7MSe8VWbGe5fiz1HZFYgqbSfS67V3T8hQ9va5uiu9U5WzDmmq1hb9H"
        PINECONE_INDEX = "semantic-scholar-v2"  
        
        BUCKET_NAME = "semantic-s3"
        FILE_KEY = "abstracts/file1.json.gz"
        
        
        processor = DocumentProcessor(
            aws_region=AWS_REGION,
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX
        )
        
        await processor.process_abstracts(
            bucket=BUCKET_NAME,
            key=FILE_KEY,
            batch_size=100
        )
        
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())