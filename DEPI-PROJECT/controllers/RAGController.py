from typing import List, Dict, Optional
import logging
from controllers.BaseController import BaseController
from stores.rag.loaders.DocumentLoader import DocumentLoader
from stores.rag.TextSplitter import TextSplitter
from stores.rag.EmbeddingsService import EmbeddingsService
from stores.rag.VectorStore import VectorStore
from stores.llm.LLMProviderFactory import LLMProviderFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class RAGController(BaseController):
    """Controller for RAG (Retrieval-Augmented Generation) operations"""

    def __init__(self, session_id=None, username=None, utility_params=None):
        super().__init__()
        self.session_id = session_id
        self.username = username
        self.utility_params = utility_params or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=self.utility_params.get('chunk_size', 1000),
            chunk_overlap=self.utility_params.get('chunk_overlap', 200),
        )
        self.embeddings_service = EmbeddingsService(config=self.app_settings)
        self.vector_store = VectorStore(
            persist_directory="./faiss_db",
            collection_name=self.utility_params.get('collection_name', 'documents')
        )

    async def index_document(self, file_path: str) -> Dict:
        """
        Index a document: load, split, embed, and store

        Args:
            file_path: Path to the document file

        Returns:
            Dict with indexing results
        """
        try:
            # Load document
            self.logger.info(f"Loading document: {file_path}")
            document = self.document_loader.load_document(file_path)

            # Split into chunks
            self.logger.info("Splitting document into chunks")
            chunks = self.text_splitter.split_text(
                text=document.content,
                metadata=document.metadata
            )

            # Convert chunks to dict format
            chunk_dicts = []
            for chunk in chunks:
                chunk_dicts.append({
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })

            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(chunk_dicts)} chunks")
            embedded_docs = self.embeddings_service.embed_documents(chunk_dicts)

            # Store in vector database
            self.logger.info("Storing chunks in vector database")
            doc_ids = self.vector_store.add_documents(embedded_docs)

            return {
                'success': True,
                'file_path': file_path,
                'num_chunks': len(chunks),
                'doc_ids': doc_ids,
                'message': f"Successfully indexed {len(chunks)} chunks from document"
            }

        except Exception as e:
            self.logger.error(f"Error indexing document: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to index document: {str(e)}"
            }

    async def index_directory(self, directory_path: str) -> Dict:
        """
        Index all documents in a directory

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Dict with indexing results
        """
        try:
            # Load all documents
            self.logger.info(f"Loading documents from directory: {directory_path}")
            documents = self.document_loader.load_directory(directory_path)

            results = []
            total_chunks = 0

            for document in documents:
                # Split into chunks
                chunks = self.text_splitter.split_text(
                    text=document.content,
                    metadata=document.metadata
                )

                # Convert to dict format
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dicts.append({
                        'content': chunk.content,
                        'metadata': chunk.metadata
                    })

                # Generate embeddings
                embedded_docs = self.embeddings_service.embed_documents(chunk_dicts)

                # Store in vector database
                doc_ids = self.vector_store.add_documents(embedded_docs)

                results.append({
                    'file_name': document.metadata.get('file_name'),
                    'num_chunks': len(chunks),
                    'doc_ids': doc_ids
                })

                total_chunks += len(chunks)

            return {
                'success': True,
                'directory_path': directory_path,
                'num_documents': len(documents),
                'total_chunks': total_chunks,
                'results': results,
                'message': f"Successfully indexed {len(documents)} documents with {total_chunks} chunks"
            }

        except Exception as e:
            self.logger.error(f"Error indexing directory: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to index directory: {str(e)}"
            }

    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            # Generate query embedding
            self.logger.info(f"Searching for: {query}")
            query_embedding = self.embeddings_service.embed_text(query, document_type="query")

            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k
            )

            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    async def rag_query(self, query: str, top_k: int = 3) -> str:
        """
        RAG query: retrieve relevant context and generate answer

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated answer
        """
        try:
            # Retrieve relevant documents
            self.logger.info(f"RAG Query: {query}")
            relevant_docs = await self.search_documents(query, top_k=top_k)

            if not relevant_docs:
                return "I couldn't find any relevant information in the knowledge base to answer your question."

            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"[Document {i}]\n{doc['content']}\n")

            context = "\n".join(context_parts)

            # Generate answer using LLM
            factory = LLMProviderFactory(config=self.app_settings)
            provider = factory.create(provider=self.app_settings.GENERATION_BACKEND)

            template_str = """
            You are a helpful AI assistant. Use the following context from the knowledge base to answer the user's question.
            If the context doesn't contain relevant information, say so politely.

            Context:
            {context}

            Question: {question}

            Instructions:
            - Provide a clear, accurate answer based on the context
            - Cite specific information from the context when possible
            - If the context doesn't fully answer the question, acknowledge the limitations
            - Keep the answer concise and focused

            Answer:
            """

            template = ChatPromptTemplate.from_template(template_str)
            provider_runnable = RunnableLambda(lambda v: provider.generate_text(v.to_string()))
            chain = template | provider_runnable

            response = chain.invoke({
                "context": context,
                "question": query
            })

            return response

        except Exception as e:
            self.logger.error(f"Error in RAG query: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    async def delete_document(self, doc_id: str) -> Dict:
        """
        Delete a document from the vector store

        Args:
            doc_id: Document ID to delete

        Returns:
            Dict with deletion result
        """
        try:
            success = self.vector_store.delete_document(doc_id)
            return {
                'success': success,
                'doc_id': doc_id,
                'message': f"Document {doc_id} deleted successfully" if success else "Failed to delete document"
            }
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Error deleting document: {str(e)}"
            }

    async def list_documents(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        List documents in the vector store

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            Dict with document list
        """
        try:
            documents = self.vector_store.list_documents(limit=limit, offset=offset)
            total_count = self.vector_store.count_documents()

            return {
                'success': True,
                'documents': documents,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents': [],
                'total_count': 0
            }

    async def get_statistics(self) -> Dict:
        """
        Get RAG system statistics

        Returns:
            Dict with system statistics
        """
        try:
            total_docs = self.vector_store.count_documents()

            return {
                'success': True,
                'total_documents': total_docs,
                'collection_name': self.vector_store.collection_name,
                'embedding_dimension': self.embeddings_service.get_embedding_dimension()
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
