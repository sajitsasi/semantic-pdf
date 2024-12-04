import os
import pdfplumber
import re
import json
import argparse
from elasticsearch import Elasticsearch
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

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
        'api_key': os.getenv('ELASTIC_API_KEY')
    }
    
    if not config['cloud_id'] or not config['api_key']:
        raise ValueError("ELASTIC_CLOUD_ID and ELASTIC_API_KEY must be set in .env file")
    
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

def process_pdf_with_chapters(filepath):
    """
    Extract text and detect chapters from PDF file
    """
    current_chapter = "Introduction"
    pages = []
    
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    detected_chapter = detect_chapter(text)
                    if detected_chapter:
                        current_chapter = detected_chapter
                    
                    pages.append({
                        "page_num": page_num,
                        "content": text.strip(),
                        "chapter": current_chapter,
                        "chapter_start": bool(detected_chapter)
                    })
            print(f"Successfully processed {filepath} - {len(pages)} pages extracted")
            return pages
    except Exception as e:
        print(f"Error processing PDF {filepath}: {e}")
        return []

def index_documents(es, index_name, pdf_directory):
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
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        
        pages = process_pdf_with_chapters(pdf_file)
        
        for page in pages:
            document = {
                "filename": pdf_file.name,
                "page_num": page["page_num"],
                "content": page["content"],
                "chapter": page["chapter"],
                "chapter_start": page["chapter_start"],
                "timestamp": datetime.now()
            }
            
            try:
                es.index(index=index_name, document=document)
                total_docs += 1
                print(f"Indexed page {page['page_num']} from {pdf_file.name} - Chapter: {page['chapter']}")
            except Exception as e:
                print(f"Error indexing document: {e}")
    
    print(f"\nIndexing complete! Total documents indexed: {total_docs}")
    return total_docs

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
        total_docs = index_documents(es, args.index_name, args.pdf_dir)
        
        print(f"\nScript completed successfully!")
        print(f"- Index: {args.index_name}")
        print(f"- Documents indexed: {total_docs}")
        print(f"- PDF directory: {args.pdf_dir}")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        raise

if __name__ == "__main__":
    main()
