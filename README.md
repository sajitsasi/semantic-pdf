# PDF Ingestion with Elasticsearch

This project provides Python scripts to process and index PDF files into Elasticsearch for semantic search and enhanced text analysis. The scripts support handling PDF content, including text, chapters, and tables, leveraging Elasticsearch's capabilities and optional integration with OpenAI's GPT models for table summarization.

---

## Features

- **PDF Text Extraction**: Extracts text from PDF files and detects chapters based on predefined patterns.
- **Table Processing**: Detects and processes tables in PDF files, with optional summarization using OpenAI's GPT models.
- **Semantic Search Indexing**: Configures Elasticsearch indices for enhanced search capabilities using BM25 and ELSER inference.
- **Configuration Driven**: Customizable index settings and mappings via `index_config.json`.
- **Integration**: Supports both Elasticsearch and OpenAI GPT for advanced data handling.

---

## Prerequisites

1. Python 3.8 or later.
2. Elasticsearch Cloud setup with API key access.
3. OpenAI API key (optional, required for table summarization).

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sajitsasi/semantic-pdf.git
   cd semantic-pdf
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:

   Create a `.env` file and add your Elasticsearch Cloud ID and API key (and optionally OpenAI API key if you want to generate embeddings for tables within the PDF):

   ```bash
   ELASTIC_CLOUD_ID=<your-cloud-id>
   ELASTIC_API_KEY=<your-api-key>
   OPENAI_API_KEY=<your-openai-api-key>
   ```

4. (Optional) Modify the `index_config.json` file to customize the index settings and mappings.

---

## Usage

1. Basic Text and Chapter Extraction:

   ```bash
   python ingest_pdf.py --pdf-dir ./pdf-files --index-name search-ml-book --index-config index_config.json
   ```

2. Text, Chapter, and Table Extraction with Table Summarization:

   ```bash
   python ingest_pdf_with_tables.py --pdf-dir ./pdf-files --index-name search-ml-book --index-config index_config.json
   ```

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
