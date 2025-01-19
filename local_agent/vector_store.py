"""
Vector store implementation with Markdown chunking and Ollama embeddings.
"""
import os
import re
import asyncio
import aiohttp
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import uuid
from .config import EmbeddingModelConfig, ConfigManager
import chromadb
from chromadb.config import Settings
import markdown
from bs4 import BeautifulSoup

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None

class MarkdownChunker:
    """Handles chunking of markdown documents with structure preservation."""
    
    def __init__(self, max_chunk_size: int = 512):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Target size for chunks in tokens (approximate)
        """
        self.max_chunk_size = max_chunk_size
        
    def _extract_headers(self, text: str) -> List[Tuple[str, int]]:
        """Extract markdown headers with their positions."""
        headers = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            headers.append((match.group(0), match.start()))
        return headers
        
    def _find_chunk_boundary(self, text: str, start: int, max_size: int) -> int:
        """
        Find a suitable chunk boundary after start position.
        Tries to break at paragraph boundaries first, then sentences.
        """
        # Look for double newline (paragraph boundary)
        text_slice = text[start:start + max_size]
        para_match = re.search(r'\n\n', text_slice)
        if para_match:
            return start + para_match.end()
            
        # Look for sentence boundary
        sentence_match = re.search(r'[.!?]\s', text_slice)
        if sentence_match:
            return start + sentence_match.end()
            
        # Fall back to word boundary
        word_match = re.search(r'\s', text_slice[::-1])
        if word_match:
            return start + max_size - word_match.start()
            
        return start + max_size
        
    def _get_chunk_context(self, text: str, chunk_start: int, headers: List[Tuple[str, int]]) -> str:
        """Get the relevant headers that apply to this chunk."""
        context = []
        for header, pos in headers:
            if pos < chunk_start:
                context = [header]  # Reset context at each header
            else:
                break
        return ' > '.join(context) if context else ''
        
    def chunk_document(self, content: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split a markdown document into chunks while preserving structure.
        
        Args:
            content: Markdown content
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        headers = self._extract_headers(content)
        
        pos = 0
        chunk_index = 0
        
        while pos < len(content):
            # Get context from headers
            context = self._get_chunk_context(content, pos, headers)
            
            # Find next chunk boundary
            chunk_size = self.max_chunk_size * 4  # Convert token target to chars
            end = self._find_chunk_boundary(content, pos, chunk_size)
            
            # Extract chunk text
            chunk_text = content[pos:end].strip()
            if not chunk_text:  # Skip empty chunks
                pos = end
                continue
                
            # Create chunk with metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "context": context,
                "source_start": pos,
                "source_end": end
            }
            
            chunk_id = str(uuid.uuid4())
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                metadata=chunk_metadata
            ))
            
            pos = end
            chunk_index += 1
            
        return chunks

# ============================================================================
# Embedding Client
# ============================================================================

class EmbeddingLLMError(Exception):
    """Base exception for EmbeddingLLMClient errors"""
    pass

class EmbeddingLLMClient:
    """LLM Client for generating embeddings."""
    
    def __init__(self, embedding_config: EmbeddingModelConfig):
        self.embedding_config = embedding_config
        
    async def generate_embeddings(self, texts: Union[str, List[str]], timeout: int = 30) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for one or more texts using nomic-embed-text.
        
        Args:
            texts: Single text string or list of text strings
            timeout: Request timeout in seconds
            
        Returns:
            List of embeddings (list of floats) for single text,
            or list of embeddings for multiple texts
            
        Raises:
            EmbeddingLLMError: If request fails or times out
        """
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            embeddings = []
            async with aiohttp.ClientSession() as session:
                for text in texts:
                    async with session.post(
                        f"{self.embedding_config.host}/api/embeddings",
                        json={
                            "model": self.embedding_config.name,
                            "prompt": text
                        },
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        embeddings.append(data["embedding"])
                        
            return embeddings[0] if len(texts) == 1 else embeddings
                
        except asyncio.TimeoutError:
            raise EmbeddingLLMError("Request to Ollama timed out")
        except aiohttp.ClientError as e:
            raise EmbeddingLLMError(f"Failed to connect to Ollama: {str(e)}")

class DocumentProcessor:
    """Handles loading and processing of different document types."""
    
    @staticmethod
    def load_markdown(file_path: Union[str, Path]) -> str:
        """
        Load and convert markdown to plain text.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Extracted plain text content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

class VectorStore:
    """Manages document embeddings and similarity search using ChromaDB."""
    
    def __init__(self, project_path: str, embedding_config: EmbeddingModelConfig, chunk_size: int = 512):
        """
        Initialize the vector store.
        
        Args:
            project_path: Base path of the project
            chunk_size: Target chunk size in tokens
        """
        self.project_path = project_path
        self.persist_dir = os.path.join(project_path, '.llama', 'vector_store')
        self.embedding_client = EmbeddingLLMClient(embedding_config)
        self.chunker = MarkdownChunker(max_chunk_size=chunk_size)
        
        # Ensure storage directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=chromadb.Settings(allow_reset=True)
        )
        
        # Create or get our collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def get_collection(self):
        """Get the current ChromaDB collection."""
        return self.collection
    
    def reset(self):
        """Reset the vector store by deleting all data and recreating the collection."""
        # Delete all collections
        self.client.reset()
        
        # Create our collection
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
    async def add_document(self, file_path: Union[str, Path], metadata: Optional[Dict] = None):
        """
        Add a document to the vector store.

        Args:
            file_path: Path to the document
            metadata: Optional metadata about the document
        """
        try:
            file_path = Path(file_path)
            base_metadata = {
                "path": str(file_path),
                "filename": file_path.name,
                **(metadata or {})
            }
            
            # Process based on file type
            if file_path.suffix.lower() == '.md':
                print(f"Reading content from: {file_path}")
                content = DocumentProcessor.load_markdown(file_path)
                print(f"Content length: {len(content)} characters")
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Chunk the document
            print("Chunking document...")
            chunks = self.chunker.chunk_document(content, base_metadata)
            print(f"Created {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            print("Generating embeddings...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_client.generate_embeddings(chunk_texts)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Add chunks to ChromaDB
            try:
                print("Adding to ChromaDB...")
                await asyncio.to_thread(
                    self.collection.add,
                    embeddings=embeddings,
                    documents=[chunk.content for chunk in chunks],
                    metadatas=[chunk.metadata for chunk in chunks],
                    ids=[chunk.id for chunk in chunks]
                )
                print("Successfully added to ChromaDB")
            except Exception as e:
                raise RuntimeError(f"Failed to add documents to ChromaDB: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to process document {file_path}: {str(e)}")
        
    async def add_documents(self, directory: Union[str, Path], metadata: Optional[Dict] = None):
        """
        Add all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            metadata: Optional metadata to apply to all documents
        """
        directory = Path(directory)
        files_processed = 0
        print(f"Scanning directory: {directory}")
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() == '.md':
                print(f"Processing file: {file_path}")
                await self.add_document(file_path, metadata)
                files_processed += 1
        
        print(f"Processed {files_processed} files")
                
    def _chunks_to_documents(self, chunks: List[Dict]) -> List[Dict]:
        """
        Convert chunk results to document results by grouping and scoring.
        
        Args:
            chunks: List of chunk results from ChromaDB
            
        Returns:
            List of document results with combined content and scores
        """
        # Group chunks by source document
        doc_chunks = {}
        for chunk in chunks:
            path = chunk['metadata']['path']
            if path not in doc_chunks:
                doc_chunks[path] = []
            doc_chunks[path].append(chunk)
            
        # Combine chunks for each document
        results = []
        for path, chunks in doc_chunks.items():
            # Sort chunks by position
            chunks.sort(key=lambda x: x['metadata']['source_start'])
            
            # Combine chunk content
            content = '\n'.join(chunk['document'] for chunk in chunks)
            
            # Use best chunk score as document score
            best_score = min(chunk['distance'] for chunk in chunks)
            
            # Get metadata from first chunk
            metadata = chunks[0]['metadata'].copy()
            # Remove chunk-specific metadata
            for key in ['chunk_index', 'context', 'source_start', 'source_end']:
                metadata.pop(key, None)
                
            results.append({
                'document': content,
                'metadata': metadata,
                'distance': best_score
            })
            
        # Sort results by score
        results.sort(key=lambda x: x['distance'])
        return results
                
    async def _preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess the query text for embedding.
        
        For longer queries, splits into chunks to match document chunking.
        For shorter queries, returns as single chunk.
        
        Args:
            query: The query text
            
        Returns:
            List of query chunks
        """
        # Clean and normalize query text
        query = query.strip()
        
        # For very short queries, no need to chunk
        if len(query) < 100:  # Approximate threshold
            return [query]
            
        # Create metadata dict for chunking
        query_metadata = {
            "type": "query",
            "original_text": query[:100] + "..." if len(query) > 100 else query
        }
        
        # Use same chunking logic as documents
        chunks = self.chunker.chunk_document(query, query_metadata)
        return [chunk.content for chunk in chunks]

    async def _generate_query_embeddings(self, query_chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for query chunks.
        
        Args:
            query_chunks: List of query text chunks
            
        Returns:
            List of embeddings for each chunk
        """
        try:
            # Generate embeddings for all chunks
            embeddings = await self.embedding_client.generate_embeddings(query_chunks)
            
            # Handle single chunk case
            if len(query_chunks) == 1:
                return [embeddings]  # Wrap single embedding in list
            return embeddings
            
        except EmbeddingLLMError as e:
            raise RuntimeError(f"Failed to generate query embeddings: {str(e)}")

    async def _search_chunks(self, query_embeddings: List[List[float]], n_results: int = 5) -> List[Dict]:
        """
        Search for similar chunks using query embeddings.
        
        Args:
            query_embeddings: List of query chunk embeddings
            n_results: Number of results to return per query embedding
            
        Returns:
            Combined and deduplicated search results
        """
        try:
            all_results = []
            
            # Search for each query embedding
            for embedding in query_embeddings:
                chunk_results = await asyncio.to_thread(
                    self.collection.query,
                    query_embeddings=[embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Format results
                for i in range(len(chunk_results['ids'][0])):
                    result = {
                        'document': chunk_results['documents'][0][i],
                        'metadata': chunk_results['metadatas'][0][i],
                        'distance': chunk_results['distances'][0][i]
                    }
                    all_results.append(result)
            
            return all_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search chunks: {str(e)}")

    async def query(self, text: str, n_results: int = 5) -> List[Dict]:
        """
        Query the vector store for similar documents.
        
        Args:
            text: Query text
            n_results: Number of results to return
            
        Returns:
            List of results with document content and metadata
        """
        try:
            # Preprocess and chunk query
            query_chunks = await self._preprocess_query(text)
            
            # Generate embeddings for query chunks
            query_embeddings = await self._generate_query_embeddings(query_chunks)
            
            # Search for similar chunks
            chunk_results = await self._search_chunks(
                query_embeddings,
                n_results=max(n_results, 3)  # Get extra results for better coverage
            )
            
            # Deduplicate and combine results
            doc_results = self._chunks_to_documents(chunk_results)
            
            # Return requested number of results
            return doc_results[:n_results]
            
        except Exception as e:
            raise RuntimeError(f"Failed to process query: {str(e)}")