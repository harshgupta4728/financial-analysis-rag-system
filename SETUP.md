# 🚀 Quick Setup Guide

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file**
   Create a `.env` file in the project root with:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional
   NEWS_API_KEY=your_news_api_key_here
   
   # Application Configuration
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   TEMPERATURE=0.7
   MAX_TOKENS=2000
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## Getting API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **News API Key**: Get from [NewsAPI](https://newsapi.org/) (optional)

## Usage

1. **Data Ingestion**: Upload PDFs and fetch market data
2. **Query & Analysis**: Ask questions about your data
3. **Market Analysis**: Get technical analysis and charts
4. **News Sentiment**: Analyze news sentiment
5. **System Info**: View configuration and statistics
