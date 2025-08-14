"""
Vector Store Module for Financial Analysis RAG System
Handles ChromaDB integration, embeddings, and retrieval operations
"""

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
import shutil
from datetime import datetime
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialVectorStore:
    """Manages vector storage and retrieval for financial documents"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store
        
        Args:
            embedding_model: Name of the sentence transformer model
            persist_directory: Directory to persist ChromaDB data
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.model = None  # Lazy load the model
        self.use_fallback = False
        self.fallback_data = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        
        # Try to initialize ChromaDB, fallback to in-memory if it fails
        self._initialize_storage()
        
        # Create or get collections
        self._initialize_collections()
    
    def _initialize_storage(self):
        """Initialize storage with fallback to in-memory"""
        try:
            # Try to use persistent ChromaDB
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"Initialized ChromaDB at {self.persist_directory}")
            self.use_fallback = False
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {str(e)}")
            logger.info("Using in-memory fallback storage")
            self._initialize_fallback_storage()
    
    def _initialize_fallback_storage(self):
        """Initialize in-memory fallback storage"""
        try:
            self.client = chromadb.Client()
            self.use_fallback = True
            logger.info("Successfully initialized in-memory ChromaDB client")
        except Exception as e:
            logger.error(f"Failed to initialize fallback storage: {str(e)}")
            raise
    
    def _load_model(self):
        """Lazy load the embedding model when needed"""
        if self.model is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model}")
                self.model = SentenceTransformer(self.embedding_model)
                logger.info(f"Successfully loaded embedding model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for different data types"""
        try:
            if self.use_fallback:
                # Use single collection for fallback mode
                self.main_collection = self.client.get_or_create_collection(
                    name="financial_data",
                    metadata={"description": "All financial data (fallback mode)"}
                )
                self.reports_collection = self.main_collection
                self.news_collection = self.main_collection
                self.market_collection = self.main_collection
                logger.info("Successfully initialized fallback collections")
            else:
                # Use separate collections for different data types
                self.main_collection = self.client.get_or_create_collection(
                    name="financial_documents",
                    metadata={"description": "Financial documents, news, and market data"}
                )
                
                self.reports_collection = self.client.get_or_create_collection(
                    name="financial_reports",
                    metadata={"description": "SEC filings and financial reports"}
                )
                
                self.news_collection = self.client.get_or_create_collection(
                    name="financial_news",
                    metadata={"description": "Financial news articles"}
                )
                
                self.market_collection = self.client.get_or_create_collection(
                    name="market_data",
                    metadata={"description": "Market data and analysis"}
                )
                
                logger.info("Successfully initialized all collections")
            
        except Exception as e:
            logger.error(f"Error initializing collections: {str(e)}")
            # If all else fails, use in-memory fallback
            logger.info("Falling back to in-memory storage")
            self._initialize_fallback_storage()
            self._initialize_collections()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            self._load_model() # Ensure model is loaded
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = "main") -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries with 'id', 'content', 'type', 'metadata'
            collection_name: Name of the collection to add to
            
        Returns:
            Success status
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False
            
            # Select collection
            if collection_name == "reports":
                collection = self.reports_collection
            elif collection_name == "news":
                collection = self.news_collection
            elif collection_name == "market":
                collection = self.market_collection
            else:
                collection = self.main_collection
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                ids.append(doc.get('id', f"doc_{len(ids)}"))
                texts.append(doc.get('content', ''))
                metadata = doc.get('metadata', {})
                metadata['type'] = doc.get('type', collection_name)
                metadata['timestamp'] = datetime.now().isoformat()
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to {collection_name} collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5, collection_name: str = "main") -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: Search query
            n_results: Number of results to return
            collection_name: Name of the collection to search in
            
        Returns:
            List of similar documents
        """
        try:
            # Select collection
            if collection_name == "reports":
                collection = self.reports_collection
            elif collection_name == "news":
                collection = self.news_collection
            elif collection_name == "market":
                collection = self.market_collection
            else:
                collection = self.main_collection
            
            # Search in collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'relevance_score': 1 - (results['distances'][0][i] if 'distances' in results else 0)
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def search_across_all_collections(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search across all collections
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents from all collections
        """
        try:
            if self.use_fallback:
                # In fallback mode, just search the main collection
                return self.search_documents(query, n_results, "main")
            
            # Search in all collections
            all_results = []
            
            collections = [
                ("reports", self.reports_collection),
                ("news", self.news_collection),
                ("market", self.market_collection),
                ("main", self.main_collection)
            ]
            
            for collection_name, collection in collections:
                try:
                    results = collection.query(
                        query_texts=[query],
                        n_results=max(1, n_results // 4)  # Distribute results across collections
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        for i in range(len(results['documents'][0])):
                            all_results.append({
                                'id': results['ids'][0][i],
                                'content': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i] if 'distances' in results else None,
                                'relevance_score': 1 - (results['distances'][0][i] if 'distances' in results else 0),
                                'type': collection_name
                            })
                except Exception as e:
                    logger.warning(f"Error searching {collection_name} collection: {str(e)}")
                    continue
            
            # Sort by relevance score and return top results
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching across all collections: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collections
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {}
            
            if self.use_fallback:
                # In fallback mode, just return main collection stats
                count = self.main_collection.count()
                stats['main'] = {
                    'document_count': count,
                    'name': 'main'
                }
            else:
                collections = {
                    "main": self.main_collection,
                    "reports": self.reports_collection,
                    "news": self.news_collection,
                    "market": self.market_collection
                }
                
                for name, collection in collections.items():
                    try:
                        count = collection.count()
                        stats[name] = {
                            'document_count': count,
                            'name': name
                        }
                    except Exception as e:
                        logger.warning(f"Error getting stats for {name}: {str(e)}")
                        stats[name] = {'document_count': 0, 'name': name}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False
    
    def reset_all_collections(self) -> bool:
        """
        Reset all collections (delete and recreate)
        
        Returns:
            Success status
        """
        try:
            if self.use_fallback:
                # In fallback mode, just reset the main collection
                try:
                    self.client.delete_collection("financial_data")
                except:
                    pass
                self._initialize_collections()
            else:
                # Delete all collections
                collection_names = ["financial_documents", "financial_reports", "financial_news", "market_data"]
                
                for name in collection_names:
                    try:
                        self.client.delete_collection(name)
                    except:
                        pass  # Collection might not exist
                
                # Reinitialize collections
                self._initialize_collections()
            
            logger.info("Successfully reset all collections")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collections: {str(e)}")
            return False
    
    def get_similar_documents(self, document_id: str, n_results: int = 5, 
                            collection_name: str = "main") -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        
        Args:
            document_id: ID of the reference document
            n_results: Number of similar documents to return
            collection_name: Name of the collection to search in
            
        Returns:
            List of similar documents
        """
        try:
            # Select collection
            if collection_name == "reports":
                collection = self.reports_collection
            elif collection_name == "news":
                collection = self.news_collection
            elif collection_name == "market":
                collection = self.market_collection
            else:
                collection = self.main_collection
            
            # Get the reference document
            results = collection.get(ids=[document_id])
            
            if not results['documents']:
                logger.warning(f"Document {document_id} not found")
                return []
            
            # Find similar documents
            similar_results = collection.query(
                query_embeddings=self.generate_embeddings([results['documents'][0]]),
                n_results=n_results + 1  # +1 to account for the original document
            )
            
            # Format and filter results
            formatted_results = []
            if similar_results['documents'] and similar_results['documents'][0]:
                for i in range(len(similar_results['documents'][0])):
                    if similar_results['ids'][0][i] != document_id:  # Exclude the original document
                        formatted_results.append({
                            'id': similar_results['ids'][0][i],
                            'content': similar_results['documents'][0][i],
                            'metadata': similar_results['metadatas'][0][i],
                            'distance': similar_results['distances'][0][i] if 'distances' in similar_results else None
                        })
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
