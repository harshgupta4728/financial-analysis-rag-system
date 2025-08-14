


"""
LLM Interface Module for Financial Analysis RAG System
Handles Gemini API integration and response generation
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialLLMInterface:
    """Interface for Gemini API and response generation"""
    
    def __init__(self, model: str = "gemini-1.5-flash-latest", temperature: float = 0.7, max_tokens: int = 2000):
        """
        Initialize the LLM interface
        
        Args:
            model: Gemini model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
        """
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_tokens
        
        # Set up Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
        )

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]], 
                         response_type: str = "analysis") -> Dict[str, Any]:
        """
        Generate a response based on query and context documents
        
        Args:
            query: User query
            context_documents: Retrieved relevant documents
            response_type: Type of response ("analysis", "summary", "risk_assessment")
            
        Returns:
            Generated response with metadata
        """
        try:
            if not context_documents:
                return {
                    'response': "I don't have enough relevant information to provide a comprehensive answer. Please try rephrasing your question or provide more specific details.",
                    'sources': [],
                    'confidence': 'low',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Prepare context
            context_text = self._prepare_context(context_documents)
            
            # Create system prompt based on response type
            system_prompt = self._create_system_prompt(response_type)
            
            # Create user prompt
            user_prompt = self._create_user_prompt(query, context_text, response_type)

            # Combine prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract response content
            response_content = response.text
            
            # Prepare sources
            sources = self._extract_sources(context_documents)
            
            # Calculate confidence based on document relevance
            confidence = self._calculate_confidence(context_documents)
            
            return {
                'response': response_content,
                'sources': sources,
                'confidence': confidence,
                'model_used': self.model_name,
                'tokens_used': 'N/A',  # Not directly available in Gemini API response
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'response': f"I encountered an error while generating the response: {str(e)}",
                'sources': [],
                'confidence': 'error',
                'timestamp': datetime.now().isoformat()
            }

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare context from retrieved documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            section = doc.get('metadata', {}).get('section', '')
            
            # Create document header
            header = f"Document {i} ({doc_type}"
            if section:
                header += f", {section}"
            header += "):"
            
            # Add content
            content = doc.get('content', '')[:1000]  # Limit content length
            
            context_parts.append(f"{header}\n{content}\n")
        
        return "\n".join(context_parts)

    def _create_system_prompt(self, response_type: str) -> str:
        """
        Create system prompt based on response type
        
        Args:
            response_type: Type of response to generate
            
        Returns:
            System prompt string
        """
        base_prompt = """You are a professional financial analyst with expertise in analyzing financial reports, market data, and news. You provide accurate, insightful, and well-reasoned analysis based on the provided context.

Key guidelines:
- Always base your analysis on the provided context documents
- Be specific and cite relevant information from the sources
- Use clear, professional language suitable for financial analysis
- If information is not available in the context, clearly state this
- Provide balanced analysis considering both positive and negative factors
- Use appropriate financial terminology and metrics
- Be concise but comprehensive in your analysis"""

        if response_type == "analysis":
            return base_prompt + """

For financial analysis, focus on:
- Key financial metrics and trends
- Risk factors and opportunities
- Market positioning and competitive analysis
- Future outlook and projections
- Investment implications"""
        
        elif response_type == "summary":
            return base_prompt + """

For summaries, focus on:
- Key highlights and main points
- Important financial figures and metrics
- Significant developments or changes
- Executive summary of findings"""
        
        elif response_type == "risk_assessment":
            return base_prompt + """

For risk assessment, focus on:
- Identified risk factors
- Risk severity and probability
- Mitigation strategies
- Impact on business operations
- Regulatory and market risks"""
        
        else:
            return base_prompt

    def _create_user_prompt(self, query: str, context: str, response_type: str) -> str:
        """
        Create user prompt with query and context
        
        Args:
            query: User's question
            context: Prepared context from documents
            response_type: Type of response requested
            
        Returns:
            User prompt string
        """
        prompt = f"""Based on the following context documents, please answer the user's question.

Context Documents:
{context}

User Question: {query}

Please provide a {response_type} that directly addresses the question using information from the provided context. If the context doesn't contain sufficient information to answer the question, clearly state this and suggest what additional information would be needed.

Format your response in a clear, structured manner with appropriate sections and bullet points where helpful."""
        
        return prompt

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents
        
        Args:
            documents: List of documents
            
        Returns:
            List of source information
        """
        sources = []
        
        for doc in documents:
            source_info = {
                'id': doc.get('id', 'unknown'),
                'type': doc.get('metadata', {}).get('type', 'unknown'),
                'timestamp': doc.get('metadata', {}).get('timestamp', 'unknown'),
                'relevance_score': 1 - (doc.get('distance', 0) if doc.get('distance') is not None else 0)
            }
            
            # Add section info for financial reports
            if doc.get('metadata', {}).get('section'):
                source_info['section'] = doc['metadata']['section']
            
            sources.append(source_info)
        
        return sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> str:
        """
        Calculate confidence level based on document relevance
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Confidence level ("high", "medium", "low")
        """
        if not documents:
            return "low"
        
        # Calculate average distance (lower is better)
        distances = [doc.get('distance', 1.0) for doc in documents if doc.get('distance') is not None]
        
        if not distances:
            return "medium"
        
        avg_distance = sum(distances) / len(distances)
        
        if avg_distance < 0.3:
            return "high"
        elif avg_distance < 0.6:
            return "medium"
        else:
            return "low"