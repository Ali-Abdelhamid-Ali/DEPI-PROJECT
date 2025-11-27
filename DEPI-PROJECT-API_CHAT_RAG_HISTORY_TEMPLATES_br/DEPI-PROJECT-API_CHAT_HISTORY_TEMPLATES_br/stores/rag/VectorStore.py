import os
import logging
from typing import List, Dict, Optional
import chromadb
import uuid


class VectorStore:
    """Vector store for storing and retrieving document embeddings using ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
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

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            self.logger.info(f"VectorStore initialized with collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error initializing VectorStore: {e}")
            raise

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
            texts = []
            metadatas = []

            for doc in documents:
                if not doc.get('embedding'):
                    self.logger.warning(f"Document missing embedding: {doc.get('metadata', {})}")
                    continue

                # Generate unique ID
                doc_id = doc.get('metadata', {}).get('id', str(uuid.uuid4()))
                ids.append(doc_id)

                # Extract data
                embeddings.append(doc['embedding'])
                texts.append(doc['content'])

                # Clean metadata (ChromaDB requires certain types)
                metadata = self._clean_metadata(doc.get('metadata', {}))
                metadatas.append(metadata)

            if not ids:
                self.logger.warning("No valid documents to add")
                return []

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

            self.logger.info(f"Added {len(ids)} documents to vector store")
            return ids

        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata to ensure compatibility with ChromaDB

        Args:
            metadata: Raw metadata dict

        Returns:
            Cleaned metadata dict
        """
        cleaned = {}
        for key, value in metadata.items():
            # ChromaDB accepts str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is None:
                cleaned[key] = ""
            else:
                # Convert other types to string
                cleaned[key] = str(value)

        return cleaned

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar documents using query embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching documents with scores
        """
        try:
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }

            # Add filters if provided
            if filter_metadata:
                query_params["where"] = filter_metadata

            # Query the collection
            results = self.collection.query(**query_params)

            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1 - results['distances'][0][i]  # Convert distance to similarity score
                    }
                    formatted_results.append(result)

            self.logger.info(f"Found {len(formatted_results)} matching documents")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            self.logger.info(f"Deleted document: {doc_id}")
            return True
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
            if filter_metadata:
                # Get matching documents first
                results = self.collection.get(where=filter_metadata)
                if results and results['ids']:
                    self.collection.delete(ids=results['ids'])
                    count = len(results['ids'])
                    self.logger.info(f"Deleted {count} documents")
                    return count
            return 0
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
            results = self.collection.get(ids=[doc_id])
            if results and results['ids'] and len(results['ids']) > 0:
                return {
                    'id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
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
            # Get all documents (ChromaDB doesn't support offset/limit directly)
            results = self.collection.get()

            documents = []
            if results and results['ids']:
                start = offset
                end = min(offset + limit, len(results['ids']))

                for i in range(start, end):
                    doc = {
                        'id': results['ids'][i],
                        'content': results['documents'][i] if results['documents'] else "",
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    documents.append(doc)

            return documents
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
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"Error counting documents: {e}")
            return 0

    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Collection {self.collection_name} reset")
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            raise