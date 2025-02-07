import boto3
from pinecone import Pinecone, ServerlessSpec
import json
import gzip
import io
import logging
from typing import List, Dict, Iterator
from tqdm import tqdm
from datetime import datetime
import ijson  # For streaming JSON parsing

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
    def __init__(self, aws_region: str, pinecone_api_key: str, pinecone_index_name: str):
        logger.info(f"Initializing DocumentProcessor with region: {aws_region}, index: {pinecone_index_name}")
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=aws_region
            )
            pc = Pinecone(api_key=pinecone_api_key)
            self.index = pc.Index(pinecone_index_name)
            logger.info("Successfully initialized clients")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings using Amazon Titan v2 with 512 dimensions and normalization
        """
        try:
            # Prepare request body
            request_body = {
                "inputText": text,
                "dimensions": 512,
                "normalize": True
            }
            
            response = self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                contentType="application/json",
                accept="*/*",
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def stream_gzipped_json_from_s3(self, bucket: str, key: str) -> Iterator[Dict]:
        """
        Stream and parse gzipped JSON Lines file from S3
        """
        logger.info(f"Starting to stream gzipped JSON Lines from s3://{bucket}/{key}")
        try:
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket, Key=key)
            
            # Stream and decompress the file
            with gzip.GzipFile(fileobj=io.BytesIO(response['Body'].read())) as gz:
                for line in gz:
                    try:
                        # Decode the line and parse JSON
                        line_str = line.decode('utf-8').strip()
                        if line_str:  # Skip empty lines
                            item = json.loads(line_str)
                            yield item
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error streaming from S3: {str(e)}")
            raise

    def process_abstracts(self, bucket: str, key: str, batch_size: int = 100):
        """
        Process JSON abstracts and store embeddings in Pinecone using streaming
        """
        try:
            logger.info(f"Starting abstract processing with batch size: {batch_size}")
            processed_count = 0
            error_count = 0
            current_batch = []
            
            # Stream and process abstracts
            for abstract in self.stream_gzipped_json_from_s3(bucket, key):
                try:
                    text_to_embed = f"{abstract.get('title', '')} {abstract.get('abstract', '')}"
                    print(abstract)
                    if text_to_embed.strip():
                        embedding = self.get_embedding(text_to_embed)
                        # Use doc_index from the abstract if available, otherwise use processed_count
                        doc_id = abstract.get('id', f'doc-{processed_count}')
                        current_batch.append({
                            'id': str(doc_id),  # Ensure ID is string
                            'values': embedding,
                            'metadata': {
                                'title': abstract.get('title', ''),
                                'abstract': abstract.get('abstract', ''),
                                'source': f"{bucket}/{key}",
                                'doc_index': doc_id,
                            }
                        })
                        processed_count += 1  # Increment count for each processed document
                        
                        # Process batch when it reaches the specified size
                        if len(current_batch) >= batch_size:
                            self.index.upsert(vectors=current_batch)
                            logger.info(f"Successfully upserted batch of {len(current_batch)} vectors. Total processed: {processed_count}")
                            current_batch = []  # Clear the batch after upsert
                            
                            if processed_count % 1000 == 0:
                                logger.info(f"Progress: {processed_count} records processed ({error_count} errors)")
                    
                except Exception as e:
                    logger.error(f"Error processing abstract: {str(e)}")
                    error_count += 1
            
            # Process any remaining items in the last batch
            if current_batch:
                self.index.upsert(vectors=current_batch)
                logger.info(f"Successfully upserted final batch of {len(current_batch)} vectors")
            
            logger.info(f"Processing complete. Total processed: {processed_count} with {error_count} errors")
            
        except Exception as e:
            logger.error(f"Fatal error during processing: {str(e)}")
            raise

def main():
    logger.info("Starting document processing")
    try:
        AWS_REGION = "us-west-2"  
        PINECONE_API_KEY = "pcsk_6aWdAr_JzZdKmXzu7MSe8VWbGe5fiz1HZFYgqbSfS67V3T8hQ9va5uiu9U5WzDmmq1hb9H"
        PINECONE_INDEX = "semantic-scholar"  
        
        BUCKET_NAME = "semantic-s3"
        FILE_KEY = "abstracts/file1.json.gz"
        
        processor = DocumentProcessor(
            aws_region=AWS_REGION,
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_index_name=PINECONE_INDEX
        )
        
        processor.process_abstracts(BUCKET_NAME, FILE_KEY)
        logger.info("Document processing completed successfully")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()