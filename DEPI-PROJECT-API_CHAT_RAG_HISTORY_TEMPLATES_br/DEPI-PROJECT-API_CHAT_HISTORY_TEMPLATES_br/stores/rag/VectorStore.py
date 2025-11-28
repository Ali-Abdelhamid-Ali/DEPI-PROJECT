import os
import json
import logging
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss
import uuid


class VectorStore:
    """Vector store for storing and retrieving document embeddings using FAISS"""

    def __init__(self, persist_directory: str = "./faiss_db", collection_name: str = "documents"):
        """
        Initialize VectorStore

        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # File paths
        self.index_path = os.path.join(persist_directory, f"{collection_name}.index")
        self.data_path = os.path.join(persist_directory, f"{collection_name}_data.pkl")

        # Initialize storage
        self.index = None
        self.documents = []  # List of {id, content, metadata}
        self.id_to_idx = {}  # Map document ID to index position
        self.dimension = None

        # Load existing data if available
        self._load()

        self.logger.info(f"VectorStore initialized with collection: {collection_name}")

    def _load(self):
        """Load existing index and data from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.data_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.dimension = data.get('dimension')
                self.logger.info(f"Loaded {len(self.documents)} documents from disk")
        except Exception as e:
            self.logger.warning(f"Could not load existing data: {e}")

    def _save(self):
        """Save index and data to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                with open(self.data_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'id_to_idx': self.id_to_idx,
                        'dimension': self.dimension
                    }, f)
                self.logger.info(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def _initialize_index(self, dimension: int):
        """Initialize FAISS index with given dimension"""
        self.dimension = dimension
        # Using IndexFlatIP for cosine similarity (normalize vectors first)
        self.index = faiss.IndexFlatIP(dimension)
        self.logger.info(f"Initialized FAISS index with dimension {dimension}")

    def add_documents(self, documents: List[Dict]) -> List[str]:
        """
        Add documents to the vector store

        Args:
            documents: List of dicts with 'content', 'embedding', and 'metadata' keys

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        try:
            ids = []
            embeddings = []
            docs_to_add = []

            for doc in documents:
                if not doc.get('embedding'):
                    self.logger.warning(f"Document missing embedding: {doc.get('metadata', {})}")
                    continue

                # Generate unique ID
                doc_id = doc.get('metadata', {}).get('id', str(uuid.uuid4()))
                ids.append(doc_id)

                # Extract embedding
                embedding = np.array(doc['embedding'], dtype=np.float32)
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)

                # Store document data
                docs_to_add.append({
                    'id': doc_id,
                    'content': doc['content'],
                    'metadata': self._clean_metadata(doc.get('metadata', {}))
                })

            if not ids:
                self.logger.warning("No valid documents to add")
                return []

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Initialize index if needed
            if self.index is None:
                self._initialize_index(embeddings_array.shape[1])

            # Add to index
            self.index.add(embeddings_array)

            # Store document data
            for i, doc in enumerate(docs_to_add):
                idx = len(self.documents)
                self.documents.append(doc)
                self.id_to_idx[doc['id']] = idx

            # Save to disk
            self._save()

            self.logger.info(f"Added {len(ids)} documents to vector store")
            return ids

        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata to ensure compatibility

        Args:
            metadata: Raw metadata dict

        Returns:
            Cleaned metadata dict
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is None:
                cleaned[key] = ""
            else:
                cleaned[key] = str(value)
        return cleaned

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar documents using query embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (basic support)

        Returns:
            List of matching documents with scores
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                self.logger.warning("Index is empty")
                return []

            # Prepare query
            query = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Search
            scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

            # Format results
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue

                doc = self.documents[idx]

                # Apply metadata filter if provided
                if filter_metadata:
                    match = all(
                        doc['metadata'].get(k) == v 
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue

                result = {
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score)
                }
                formatted_results.append(result)

            self.logger.info(f"Found {len(formatted_results)} matching documents")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID (marks as deleted, requires rebuild for actual removal)

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                # Mark as deleted by clearing content
                self.documents[idx] = {
                    'id': doc_id,
                    'content': '',
                    'metadata': {'deleted': True}
                }
                del self.id_to_idx[doc_id]
                self._save()
                self.logger.info(f"Marked document as deleted: {doc_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def delete_documents(self, filter_metadata: Dict = None) -> int:
        """
        Delete documents matching filter criteria

        Args:
            filter_metadata: Metadata filters

        Returns:
            Number of documents deleted
        """
        try:
            count = 0
            if filter_metadata:
                for doc_id, idx in list(self.id_to_idx.items()):
                    doc = self.documents[idx]
                    match = all(
                        doc['metadata'].get(k) == v 
                        for k, v in filter_metadata.items()
                    )
                    if match:
                        self.delete_document(doc_id)
                        count += 1
            self.logger.info(f"Deleted {count} documents")
            return count
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return 0

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Get a document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None
        """
        try:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                doc = self.documents[idx]
                if not doc['metadata'].get('deleted'):
                    return {
                        'id': doc['id'],
                        'content': doc['content'],
                        'metadata': doc['metadata']
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id}: {e}")
            return None

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List all documents in the collection

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of documents
        """
        try:
            # Filter out deleted documents
            active_docs = [
                doc for doc in self.documents 
                if not doc['metadata'].get('deleted')
            ]
            
            # Apply pagination
            start = offset
            end = min(offset + limit, len(active_docs))
            
            return [
                {
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata']
                }
                for doc in active_docs[start:end]
            ]
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []

    def count_documents(self) -> int:
        """
        Get total number of documents in the collection

        Returns:
            Document count
        """
        try:
            return len([
                doc for doc in self.documents 
                if not doc['metadata'].get('deleted')
            ])
        except Exception as e:
            self.logger.error(f"Error counting documents: {e}")
            return 0

    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.index = None
            self.documents = []
            self.id_to_idx = {}
            self.dimension = None
            
            # Delete files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.data_path):
                os.remove(self.data_path)
                
            self.logger.info(f"Collection {self.collection_name} reset")
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            raise