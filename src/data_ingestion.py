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
        self.ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
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
        Fetch financial news using Alpha Vantage for better reliability on deployed apps.
        """
        if not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in your secrets.")
            return []

        news_articles = []
        try:
            # Sahi URL aur parameters Alpha Vantage ke liye
            ticker_query = f"tickers={ticker}" if ticker else "topics=financial-markets"
            
            url = (f"https://www.alphavantage.co/query?"
                   f"function=NEWS_SENTIMENT&"
                   f"{ticker_query}&"
                   f"limit=100&"
                   f"apikey={self.alpha_vantage_api_key}") # Yahan 'alpha_vantage_api_key' variable use ho raha hai

            response = requests.get(url)
            data = response.json()

            if "feed" not in data:
                logger.error(f"Failed to fetch news from Alpha Vantage. Response: {data}")
                return []

            for entry in data.get("feed", []):
                article = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('url', ''),
                    'published_date': entry.get('time_published', ''),
                    'source': entry.get('source', 'Unknown'),
                    'full_text': entry.get('summary', '')
                }
                news_articles.append(article)
            
            logger.info(f"Successfully fetched {len(news_articles)} news articles from Alpha Vantage")
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching financial news from Alpha Vantage: {str(e)}")
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