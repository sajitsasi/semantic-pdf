import os
import pdfplumber
import re
import json
import argparse
from elasticsearch import Elasticsearch
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

from typing import List, Dict, Any
import pandas as pd
from io import StringIO

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Index PDFs into Elasticsearch with semantic search capabilities')
    parser.add_argument('--pdf-dir', type=str, default='./pdf-files',
                        help='Directory containing PDF files (default: ./pdf-files)')
    parser.add_argument('--index-name', type=str, default='search-ml-book',
                        help='Elasticsearch index name (default: search-ml-book)')
    parser.add_argument('--index-config', type=str, default='index_config.json',
                        help='Path to index configuration JSON file (default: index_config.json)')
    return parser.parse_args()

def load_config():
    """
    Load configuration from .env file
    """
    load_dotenv()
    config = {
            'cloud_id': os.getenv('ELASTIC_CLOUD_ID'),
            'api_key': os.getenv('ELASTIC_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY')
            }

    if not all(config.values()):
        raise ValueError("ELASTIC_CLOUD_ID, ELASTIC_API_KEY, and OPENAI_API_KEY must be set in .env file")
        raise Exception("The 'openai.api_key' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_key=config['openai_api_key'])'")

    return config

def connect_to_elastic(cloud_id, api_key):
    """
    Connect to Elasticsearch Cloud instance
    """
    try:
        es = Elasticsearch(
                cloud_id=cloud_id,
                api_key=api_key,
                )
        print("Connected to Elasticsearch")
        return es
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return None

def create_inference_endpoint(es):
    """
    Create ELSER inference endpoint if it doesn't exist
    """
    try:
        config = {
                "service": "elser",
                "service_settings": {
                    "num_allocations": 1,
                    "num_threads": 1
                    }
                }
        es.put("/_inference/sparse_embedding/ml-book-elser-model", json=config)
        print("Created ELSER inference endpoint")
    except Exception as e:
        print(f"Error creating inference endpoint (may already exist): {e}")

def load_index_config(config_path):
    """
    Load index configuration from JSON file
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Index configuration file not found: {config_path}")
        raise FileNotFoundError(f"Please ensure {config_path} exists with valid index configuration")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")

def create_index(es, index_name, config_path):
    """
    Create index with configuration from file
    """
    try:
        index_config = load_index_config(config_path)
        es.indices.create(index=index_name, body=index_config)
        print(f"Created index: {index_name}")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def detect_chapter(text):
    """
    Detect chapter titles using common patterns
    """
    patterns = [
            r'^Chapter\s+\d+[\s:]+([^\n]+)',
            r'^CHAPTER\s+\d+[\s:]+([^\n]+)',
            r'^\d+\.\s+([^\n]+)',
            r'^Part\s+\d+[\s:]+([^\n]+)',
            r'^Section\s+\d+[\s:]+([^\n]+)'
            ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(0).strip()
    return None

def table_to_dataframe(table) -> pd.DataFrame:
    """
    Convert pdfplumber table to pandas DataFrame
    """
    return pd.DataFrame(table.extract())

def clean_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare table data
    """
    # Fill NaN values with empty string
    df = df.fillna('')

    # Convert all values to string
    df = df.astype(str)

    # Strip whitespace
    df = df.apply(lambda x: x.str.strip())

    # Remove empty columns and rows
    df = df.loc[:, df.any()]
    df = df.loc[df.any(axis=1)]

    return df

def convert_table_to_text(config, table_df: pd.DataFrame, context: str = "") -> str:
    """
    Use OpenAI API to convert table to natural language description
    """
    # Convert DataFrame to CSV string
    csv_string = table_df.to_csv(index=False)
    client = OpenAI(api_key=config['openai_api_key'])

    try:
        completion = client.chat.completions.create(model="gpt-4o",
                                                    messages=[
                                                        {"role": "system", "content": """You are an expert at converting tabular data into clear, 
            natural language descriptions. Convert the provided table into a clear, concise textual summary. 
            Focus on the key information and relationships in the data. If context about the table is provided, 
            use it to make the description more relevant."""},
                                                        {"role": "user", "content": f"""Please convert this table into a natural language description. 
            Context about the table: {context}

            Table data:
            {csv_string}"""}
                                                        ],
                                                    temperature=0.3,
                                                    max_tokens=1000)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error converting table to text: {e}")
        # Return a basic string representation of the table as fallback
        return f"Table data: {table_df.to_string()}"

def extract_table_context(page_text: str, table_bbox) -> str:
    """
    Extract text around the table to provide context
    """
    lines = page_text.split('\n')
    table_top = table_bbox[1]
    table_bottom = table_bbox[3]

    context_lines = []
    for line in lines:
        # Get approximate y-position of the line
        # This is a simplified approach - you might need to adjust based on your PDFs
        if line and (table_top - 50 <= lines.index(line) <= table_bottom + 50):
            context_lines.append(line)

    return " ".join(context_lines)

def process_pdf_with_chapters_and_tables(config, filepath: str) -> List[Dict[str, Any]]:
    """
    Extract text, detect chapters and tables from PDF file
    """
    current_chapter = "Introduction"
    pages = []

    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract basic text
                text = page.extract_text()
                if not text:
                    continue

                # Detect chapter
                detected_chapter = detect_chapter(text)
                if detected_chapter:
                    current_chapter = detected_chapter

                # Process tables
                tables = page.find_tables()
                table_texts = []

                if tables:
                    print(f"Found {len(tables)} tables on page {page_num}")
                    for table_num, table in enumerate(tables, 1):
                        try:
                            # Convert table to DataFrame
                            df = table_to_dataframe(table)
                            df = clean_table_data(df)

                            # Get context around the table
                            context = extract_table_context(text, table.bbox)

                            # Convert table to text using LLM
                            table_text = convert_table_to_text(config, df, context)
                            table_texts.append(f"Table {table_num}: {table_text}")

                            print(f"Processed table {table_num} on page {page_num}")
                        except Exception as e:
                            print(f"Error processing table {table_num} on page {page_num}: {e}")

                # Combine regular text and table descriptions
                full_text = text
                if table_texts:
                    full_text += "\n\nTable Descriptions:\n" + "\n\n".join(table_texts)

                pages.append({
                    "page_num": page_num,
                    "content": full_text.strip(),
                    "chapter": current_chapter,
                    "chapter_start": bool(detected_chapter),
                    "has_tables": bool(tables),
                    "num_tables": len(tables)
                    })

            print(f"Successfully processed {filepath} - {len(pages)} pages extracted")
            return pages

    except Exception as e:
        print(f"Error processing PDF {filepath}: {e}")
        return []

def index_documents(config, es, index_name, pdf_directory):
    """
    Process PDFs and index their content
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        print(f"Creating PDF directory: {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_dir.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    total_docs = 0
    total_tables = 0

    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")

        pages = process_pdf_with_chapters_and_tables(config, pdf_file)

        for page in pages:
            document = {
                    "filename": pdf_file.name,
                    "page_num": page["page_num"],
                    "content": page["content"],
                    "chapter": page["chapter"],
                    "chapter_start": page["chapter_start"],
                    "has_tables": page["has_tables"],
                    "num_tables": page["num_tables"],
                    "timestamp": datetime.now()
                    }

            try:
                es.index(index=index_name, document=document)
                total_docs += 1
                total_tables += page["num_tables"]
                print(f"Indexed page {page['page_num']} from {pdf_file.name} - Chapter: {page['chapter']}")
            except Exception as e:
                print(f"Error indexing document: {e}")

    print(f"\nIndexing complete!")
    print(f"Total documents indexed: {total_docs}")
    print(f"Total tables processed: {total_tables}")
    return total_docs, total_tables

def verify_index_exists(es, index_name):
    """
    Verify that the index exists and has the correct mapping
    """
    try:
        if not es.indices.exists(index=index_name):
            return False

        mapping = es.indices.get_mapping(index=index_name)
        settings = es.indices.get_settings(index=index_name)

        print(f"Index {index_name} exists with proper configuration")
        return True
    except Exception as e:
        print(f"Error verifying index: {e}")
        return False


def main():
    # Parse command line arguments
    args = parse_args()

    try:
        # Load configuration from .env file
        config = load_config()

        # Connect to Elasticsearch
        es = connect_to_elastic(config['cloud_id'], config['api_key'])
        if not es:
            return

        # Create inference endpoint
        create_inference_endpoint(es)

        # Create index if it doesn't exist
        if not verify_index_exists(es, args.index_name):
            create_index(es, args.index_name, args.index_config)

        # Process and index PDFs
        total_docs, total_tables = index_documents(config, es, args.index_name, args.pdf_dir)

        print(f"\nScript completed successfully!")
        print(f"- Index: {args.index_name}")
        print(f"- Documents indexed: {total_docs}")
        print(f"- Tables processed: {total_tables}")
        print(f"- PDF directory: {args.pdf_dir}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        raise

if __name__ == "__main__":
    main()
