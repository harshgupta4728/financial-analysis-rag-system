"""
Data Ingestion Module for Financial Analysis RAG System
Handles loading and processing of financial reports, market data, and news
"""

import os
import re
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import PyPDF2
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataIngestion:
    """Handles ingestion of financial data from multiple sources"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF financial reports
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)  # Remove special characters
            
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return ""
    
    def get_market_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch market data for a given ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary containing market data and metadata
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist_data = stock.history(period=period)
            
            # Get basic info
            info = stock.info
            
            # Calculate additional metrics
            if not hist_data.empty:
                hist_data['Daily_Return'] = hist_data['Close'].pct_change()
                hist_data['Volatility'] = hist_data['Daily_Return'].rolling(window=20).std()
                hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
                hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            
            market_data = {
                'ticker': ticker,
                'historical_data': hist_data,
                'info': info,
                'current_price': hist_data['Close'].iloc[-1] if not hist_data.empty else None,
                'volume': hist_data['Volume'].iloc[-1] if not hist_data.empty else None,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully fetched market data for {ticker}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {str(e)}")
            return {}
    
    def get_financial_news(self, ticker: str = None, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch financial news articles using NewsAPI for reliability.
        
        Args:
            ticker: Optional stock ticker to filter news.
            days_back: Number of days to look back for news.
            
        Returns:
            List of news articles with metadata.
        """
        if not self.news_api_key:
            logger.warning("News API key not found. Please set NEWS_API_KEY in your environment.")
            return []

        news_articles = []
        
        try:
            # Calculate the 'from' date for the API query
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Construct the API URL
            # We search for the ticker in top business headlines in the US
            if ticker:
                query = f"q={ticker}&"
            else:
                query = "category=business&" # General business news if no ticker
                
            url = (f"https://newsapi.org/v2/top-headlines?"
                   f"{query}"
                   f"from={from_date}&"
                   "sortBy=popularity&"
                   "language=en&"
                   f"apiKey={self.news_api_key}")

            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch news from NewsAPI. Status code: {response.status_code}")
                return []
                
            data = response.json()
            
            for entry in data.get("articles", []):
                # Format the article to match the expected structure
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('description', ''),
                    'link': entry.get('url', ''),
                    'published_date': entry.get('publishedAt', ''),
                    'source': entry.get('source', {}).get('name', 'Unknown'),
                    'full_text': entry.get('content', entry.get('description', '')) # Fallback to description
                }
                news_articles.append(article)
            
            logger.info(f"Successfully fetched {len(news_articles)} news articles from NewsAPI")
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching financial news: {str(e)}")
            return []
    
    def get_company_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Get company financial statements and ratios
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing financial data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get key metrics
            financials = {
                'ticker': ticker,
                'income_statement': income_stmt.to_dict() if income_stmt is not None else {},
                'balance_sheet': balance_sheet.to_dict() if balance_sheet is not None else {},
                'cash_flow': cash_flow.to_dict() if cash_flow is not None else {},
                'key_metrics': stock.info,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully fetched financials for {ticker}")
            return financials
            
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {str(e)}")
            return {}
    
    def process_uploaded_pdf(self, uploaded_file) -> str:
        """
        Process uploaded PDF file from Streamlit
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
            
            logger.info("Successfully processed uploaded PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error processing uploaded PDF: {str(e)}")
            return ""