import asyncio
import boto3
import logging
from datetime import datetime
from typing import List, Optional
from parquet_creator import ParquetCreator
import os 
import random
import traceback
import sys
from botocore.exceptions import ClientError

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
logger.info("AWS Environment Variables:")
logger.info(f"AWS_ACCESS_KEY_ID present: {'AWS_ACCESS_KEY_ID' in os.environ}")
logger.info(f"AWS_SECRET_ACCESS_KEY present: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
logger.info(f"AWS_REGION present: {'AWS_REGION' in os.environ}")

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Handle any uncaught exceptions and log them properly"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_uncaught_exception

class WaveProcessor:
    def __init__(self, 
                 aws_region: str,
                 source_bucket: str,
                 files_prefix: str = "abstracts/",
                 wave_size: int = 4,
                 embeddings_per_instance: int = 40):
        """Initialize the WaveProcessor with AWS configuration and processing parameters"""
        try:
            logger.info(f"Initializing WaveProcessor with config: region={aws_region}, "
                       f"bucket={source_bucket}, prefix={files_prefix}, "
                       f"wave_size={wave_size}, embeddings={embeddings_per_instance}")
            
            self.aws_region = aws_region
            self.source_bucket = source_bucket
            self.files_prefix = files_prefix
            self.wave_size = wave_size
            self.embeddings_per_instance = embeddings_per_instance
            
            # Initialize S3 client with configured region
            self.s3 = boto3.client('s3', region_name=aws_region)
            logger.info("Successfully initialized S3 client")
            
        except Exception as e:
            logger.critical(f"Failed to initialize WaveProcessor: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def list_input_files(self) -> List[str]:
        """List all .json.gz files in the input bucket"""
        try:
            logger.info(f"Starting to list files from bucket: {self.source_bucket} with prefix: {self.files_prefix}")
            files = []
            page_count = 0
            
            try:
                paginator = self.s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.source_bucket, Prefix=self.files_prefix)
                
                for page in pages:
                    page_count += 1
                    logger.debug(f"Processing page {page_count} of S3 listing")
                    
                    if 'Contents' not in page:
                        logger.warning(f"No 'Contents' found in page {page_count}")
                        continue
                    
                    page_files = [obj['Key'] for obj in page['Contents'] 
                                if obj['Key'].endswith('.json.gz')]
                    
                    if page_files:
                        logger.debug(f"Found {len(page_files)} files in page {page_count}")
                        files.extend(page_files)
                
            except ClientError as e:
                logger.error(f"AWS error during file listing: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during file listing: {str(e)}")
                raise
            
            if not files:
                logger.warning(f"No .json.gz files found in bucket {self.source_bucket}")
            else:
                logger.info(f"Found {len(files)} files to process")
                logger.debug(f"First 5 files: {files[:5]}")
            
            return files
            
        except Exception as e:
            logger.error(f"Error in list_input_files: {str(e)}\n{traceback.format_exc()}")
            raise

    def chunk_files_into_waves(self, files: List[str]) -> List[List[str]]:
        """Split files into waves of specified size"""
        try:
            if not files:
                logger.warning("No files provided to chunk into waves")
                return []
            
            logger.info(f"Chunking {len(files)} files into waves of size {self.wave_size}")
            waves = [files[i:i + self.wave_size] 
                    for i in range(0, len(files), self.wave_size)]
            
            # Log wave distribution
            for i, wave in enumerate(waves, 1):
                logger.debug(f"Wave {i} contains {len(wave)} files")
            
            logger.info(f"Created {len(waves)} waves")
            return waves
            
        except Exception as e:
            logger.error(f"Error chunking files into waves: {str(e)}\n{traceback.format_exc()}")
            raise
    
    async def process_single_file(self, file_key: str) -> None:
        """Process a single file using ParquetCreator with exponential backoff"""
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting processing of file: {file_key} (attempt {attempt + 1})")
                
                creator = ParquetCreator(
                    aws_region=self.aws_region,
                    max_concurrent_embeddings=self.embeddings_per_instance
                )
                
                await creator.process_abstracts(
                    bucket=self.source_bucket,
                    key=file_key,
                    batch_size=5000
                )
                
                logger.info(f"Successfully completed processing file: {file_key}")
                return
                
            except Exception as e:
                if "ThrottlingException" in str(e):
                    if attempt < max_retries - 1:
                        delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                        logger.warning(f"Throttling detected for {file_key}, "
                                     f"waiting {delay:.2f} seconds before retry")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries reached for {file_key}: {str(e)}")
                        raise
                else:
                    logger.error(f"Error processing {file_key}: {str(e)}\n{traceback.format_exc()}")
                    raise
    
    async def process_wave(self, wave_files: List[str], wave_num: int) -> None:
        """Process a wave of files concurrently"""
        try:
            logger.info(f"Starting wave {wave_num} processing with {len(wave_files)} files")
            tasks = []
            
            for file_key in wave_files:
                tasks.append(self.process_single_file(file_key))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results and log outcomes
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count
            
            logger.info(f"Wave {wave_num} completed - "
                       f"Successes: {success_count}, Failures: {failure_count}")
            
            # Handle any failures
            for file_key, result in zip(wave_files, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {file_key} in wave {wave_num}: {str(result)}")
                    if not "ThrottlingException" in str(result):
                        raise result
            
        except Exception as e:
            logger.error(f"Error in wave {wave_num}: {str(e)}\n{traceback.format_exc()}")
            raise
    
    async def process_all_waves(self) -> None:
        """Process all files in waves with delays between waves"""
        try:
            logger.info("Starting process_all_waves")
            all_files = self.list_input_files()
            
            if not all_files:
                logger.warning("No files found to process")
                return
            
            waves = self.chunk_files_into_waves(all_files)
            logger.info(f"Processing {len(all_files)} files in {len(waves)} waves")
            
            for wave_num, wave_files in enumerate(waves, 1):
                try:
                    logger.info(f"Starting wave {wave_num}/{len(waves)}")
                    await self.process_wave(wave_files, wave_num)
                    
                    if wave_num < len(waves):
                        logger.info(f"Waiting 3 seconds before starting wave {wave_num + 1}")
                        await asyncio.sleep(3)
                
                except Exception as e:
                    logger.error(f"Error in wave {wave_num}: {str(e)}")
                    raise
            
            logger.info("Successfully completed all waves")
            
        except Exception as e:
            logger.error(f"Error in process_all_waves: {str(e)}\n{traceback.format_exc()}")
            raise

async def main():
    """Main function to run the wave processor"""
    try:
        logger.info("Starting main process")
        
        processor = WaveProcessor(
            aws_region="us-west-2",
            source_bucket="semantic-s3",
            wave_size=5,
            embeddings_per_instance=100
        )
        
        await processor.process_all_waves()
        logger.info("Main process completed successfully")
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        logger.info("Main process exiting")
        await asyncio.sleep(2)  # Ensure logs are written

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error in main thread: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)