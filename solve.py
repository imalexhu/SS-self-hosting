import pandas as pd
import json
import pyarrow as pa

# Read the current parquet file
location="s3://semantic-s3/semantic_data/merged/file1/final.parquet"
df = pd.read_parquet(location)

# Convert the structured metadata columns into a single JSON string
def combine_metadata(row):
    metadata_dict = {
        'abstract': row['abstract'],
        'doc_index': row['doc_index'],
        'source': row['source'],
        'title': row['title']
    }
    # Convert to JSON string
    return json.dumps(metadata_dict)

# Create new dataframe with correct format
df_formatted = pd.DataFrame({
    'id': df['id'].astype(str),  # Ensure id is string
    'values': df['values'],      # Your vector values column
    'metadata': df.apply(combine_metadata, axis=1)  # Combined JSON string
})

# Save to a new location with proper namespace structure
df_formatted.to_parquet(location)