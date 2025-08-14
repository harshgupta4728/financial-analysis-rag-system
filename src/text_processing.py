"""
Text Processing Module for Financial Analysis RAG System
Handles chunking, cleaning, and preprocessing of financial text data
"""

import re
import nltk
from typing import List, Dict, Any
import logging
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialTextProcessor:
    """Handles text processing and chunking for financial documents"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stop_words = set(stopwords.words('english'))

        # Financial-specific stop words
        financial_stop_words = {
            'page', 'section', 'table', 'figure', 'note', 'notes',
            'million', 'billion', 'thousand', 'dollars', 'percent',
            'fiscal', 'quarter', 'year', 'annual', 'quarterly'
        }
        self.stop_words.update(financial_stop_words)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text input

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\$\%]', '', text)

        # Normalize numbers and percentages
        text = re.sub(r'(\d+),(\d{3})', r'\1\2', text)  # Remove commas in numbers
        text = re.sub(r'(\d+)%', r'\1 percent', text)  # Normalize percentages

        # Remove page numbers and headers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        return text.strip()

    def extract_financial_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key financial sections from reports

        Args:
            text: Full report text

        Returns:
            Dictionary of section names and their content
        """
        sections = {}

        # Common financial report sections
        section_patterns = {
            'management_discussion': r'(Management[\'s]* Discussion|MD&A).*?(?=Item|$|Risk Factors)',
            'risk_factors': r'(Risk Factors|Risk Management).*?(?=Item|$|Management)',
            'financial_statements': r'(Financial Statements|Consolidated Statements).*?(?=Item|$|Management)',
            'business_overview': r'(Business|Company Overview|Description of Business).*?(?=Item|$|Management)',
            'executive_summary': r'(Executive Summary|Summary|Overview).*?(?=Item|$|Management)'
        }

        for section_name, pattern in section_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                sections[section_name] = self.clean_text(matches[0])

        return sections

    # def chunk_text(self, text: str, chunk_type: str = "general") -> List[Dict[str, Any]]:
    #     """
    #     Create chunks from text based on type

    #     Args:
    #         text: Text to chunk
    #         chunk_type: Type of text ("financial_report", "news", "market_data")

    #     Returns:
    #         List of chunks with metadata
    #     """
    #     chunks = []

    #     if chunk_type == "financial_report":
    #         chunks = self._chunk_financial_report(text)
    #     elif chunk_type == "news":
    #         chunks = self._chunk_news_article(text)
    #     elif chunk_type == "market_data":
    #         chunks = self._chunk_market_data(text)
    #     else:
    #         chunks = self._chunk_general_text(text)

    #     return chunks
    def chunk_text(self, text: str, chunk_type: str, doc_id_prefix: str) -> List[Dict[str, Any]]:
        """
        Create chunks from text with a unique document prefix for IDs.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"{doc_id_prefix}_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': chunk_type,
                        'metadata': {
                            'chunk_size': len(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    # Apply overlap
                    overlap_sentences = sent_tokenize(current_chunk)[-3:]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                else:
                    # Sentence itself is longer than chunk size
                    chunks.append({
                        'id': f"{doc_id_prefix}_{chunk_id}",
                        'content': sentence[:self.chunk_size],
                        'type': chunk_type,
                        'metadata': {
                            'chunk_size': len(sentence[:self.chunk_size]),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    current_chunk = ""
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc_id_prefix}_{chunk_id}",
                'content': current_chunk.strip(),
                'type': chunk_type,
                'metadata': {
                    'chunk_size': len(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
            })

        return chunks

    # --- START OF THE FIX ---
    # This function is now simplified to be more robust for any PDF.
    def _chunk_financial_report(self, text: str) -> List[Dict[str, Any]]:
        """Chunk financial reports by paragraphs"""
        chunks = []

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"financial_report_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': 'financial_report',
                        'metadata': {
                            'chunk_size': len(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = sent_tokenize(paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) > self.chunk_size:
                            if temp_chunk:
                                chunks.append({
                                    'id': f"financial_report_{chunk_id}",
                                    'content': temp_chunk.strip(),
                                    'type': 'financial_report',
                                    'metadata': {
                                        'chunk_size': len(temp_chunk),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                })
                                chunk_id += 1
                                temp_chunk = sentence
                            else:
                                chunks.append({
                                    'id': f"financial_report_{chunk_id}",
                                    'content': sentence[:self.chunk_size],
                                    'type': 'financial_report',
                                    'metadata': {
                                        'chunk_size': len(sentence[:self.chunk_size]),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                })
                                chunk_id += 1
                        else:
                            temp_chunk += " " + sentence
                    current_chunk = temp_chunk
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"financial_report_{chunk_id}",
                'content': current_chunk.strip(),
                'type': 'financial_report',
                'metadata': {
                    'chunk_size': len(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
            })

        return chunks
    # --- END OF THE FIX ---

    def _chunk_news_article(self, text: str) -> List[Dict[str, Any]]:
        """Chunk news articles by paragraphs"""
        chunks = []

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"news_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': 'news',
                        'metadata': {
                            'chunk_size': len(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = sent_tokenize(paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) > self.chunk_size:
                            if temp_chunk:
                                chunks.append({
                                    'id': f"news_{chunk_id}",
                                    'content': temp_chunk.strip(),
                                    'type': 'news',
                                    'metadata': {
                                        'chunk_size': len(temp_chunk),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                })
                                chunk_id += 1
                                temp_chunk = sentence
                            else:
                                chunks.append({
                                    'id': f"news_{chunk_id}",
                                    'content': sentence[:self.chunk_size],
                                    'type': 'news',
                                    'metadata': {
                                        'chunk_size': len(sentence[:self.chunk_size]),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                })
                                chunk_id += 1
                        else:
                            temp_chunk += " " + sentence
                    current_chunk = temp_chunk
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"news_{chunk_id}",
                'content': current_chunk.strip(),
                'type': 'news',
                'metadata': {
                    'chunk_size': len(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
            })

        return chunks

    def _chunk_market_data(self, text: str) -> List[Dict[str, Any]]:
        """Chunk market data text"""
        # Market data is usually structured, so we'll chunk by sentences
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"market_data_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': 'market_data',
                        'metadata': {
                            'chunk_size': len(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    current_chunk = sentence
                else:
                    chunks.append({
                        'id': f"market_data_{chunk_id}",
                        'content': sentence[:self.chunk_size],
                        'type': 'market_data',
                        'metadata': {
                            'chunk_size': len(sentence[:self.chunk_size]),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"market_data_{chunk_id}",
                'content': current_chunk.strip(),
                'type': 'market_data',
                'metadata': {
                    'chunk_size': len(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
            })

        return chunks

    def _chunk_general_text(self, text: str) -> List[Dict[str, Any]]:
        """General text chunking by sentences"""
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'id': f"general_{chunk_id}",
                        'content': current_chunk.strip(),
                        'type': 'general',
                        'metadata': {
                            'chunk_size': len(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
                    current_chunk = sentence
                else:
                    chunks.append({
                        'id': f"general_{chunk_id}",
                        'content': sentence[:self.chunk_size],
                        'type': 'general',
                        'metadata': {
                            'chunk_size': len(sentence[:self.chunk_size]),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'id': f"general_{chunk_id}",
                'content': current_chunk.strip(),
                'type': 'general',
                'metadata': {
                    'chunk_size': len(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
            })

        return chunks

    def extract_key_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract key financial metrics from text

        Args:
            text: Text to analyze

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        # Revenue patterns
        revenue_patterns = [
            r'revenue.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)',
            r'total revenue.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)',
            r'net revenue.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)'
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['revenue'] = matches[0]
                break

        # Profit patterns
        profit_patterns = [
            r'net income.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)',
            r'net profit.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)',
            r'earnings.*?(\$[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand)?)'
        ]

        for pattern in profit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['net_income'] = matches[0]
                break

        # Growth patterns
        growth_patterns = [
            r'growth.*?(\d+(?:\.\d+)?\s*%)',
            r'increase.*?(\d+(?:\.\d+)?\s*%)',
            r'decrease.*?(\d+(?:\.\d+)?\s*%)'
        ]

        for pattern in growth_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics['growth_rate'] = matches[0]
                break

        return metrics