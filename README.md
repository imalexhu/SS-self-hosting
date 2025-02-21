# Parquet Creator Setup Guide

## Prerequisites
- Python 3.x
- AWS credentials with appropriate permissions
- Access to the source JSON files

## Setup Instructions

1. **Create Environment File**
   ```bash
   # Copy the example environment file
   cp env.example .env
   ```

2. **Configure AWS Credentials**
   Edit `.env` and add your AWS credentials:
   ```plaintext
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   ```

3. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   source venv/bin/activate  # On Unix/macOS
   # OR
   .\venv\Scripts\activate  # On Windows
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure File Key**
   - Open `parquet_creator.py`
   - Locate the `filekey` variable
   - Update it with your desired file key from the available options (e.g., `abstracts/file3.json.gz`)

6. **Run the Script**
   ```bash
   python3 parquet_creator.py
   ```

## Available File Keys
File keys range from `abstracts/file3.json.gz` to `abstracts/file60.json.gz`. Choose the appropriate file key based on your needs.

You can find the complete list of available file keys in this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1kpellHWIgeqTURg2fC7amYvTDo4lPc17cfNexkEE0c0/edit?usp=sharing).

## Notes
- Ensure your AWS credentials have the necessary permissions
- The virtual environment should be activated before running the script
- Check the console output for any error messages during execution