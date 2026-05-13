"""
Vector Database Integration using ChromaDB
Manages embedding and retrieval of clinical and surgical knowledge
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from medsync_ai.utils.logger import get_logger

# Handle chromadb import with Python 3.14 compatibility
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception as e:
    CHROMADB_AVAILABLE = False
    chromadb = None
from medsync_ai.utils.text_processor import clean_text, chunk_text
from medsync_ai.config.settings import (
    CHROMADB_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    CHROMADB_COLLECTION_CLINICAL,
    CHROMADB_COLLECTION_SURGICAL,
    CHROMADB_COLLECTION_VQLA,
    CHUNK_SIZE
)

logger = get_logger(__name__)

class EmbeddingEngine:
    """Handles text embedding using sentence-transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed single text string
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        text = clean_text(text)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts efficiently
        
        Args:
            texts: List of texts
        
        Returns:
            Array of embeddings
        """
        texts = [clean_text(t) for t in texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True, 
                                       show_progress_bar=True)
        return embeddings
    
    def embed_document(self, text: str, chunk_size: int = CHUNK_SIZE) -> List[Tuple[str, np.ndarray]]:
        """
        Embed document by chunks
        
        Args:
            text: Full document text
            chunk_size: Characters per chunk
        
        Returns:
            List of (chunk, embedding) tuples
        """
        chunks = chunk_text(text, chunk_size)
        embeddings = self.embed_texts(chunks)
        
        return [(chunk, emb) for chunk, emb in zip(chunks, embeddings)]


class VectorDatabase:
    """ChromaDB wrapper for knowledge retrieval"""
    
    def __init__(self, db_path: Path = CHROMADB_DIR, 
                 embedding_engine: Optional[EmbeddingEngine] = None):
        """
        Initialize vector database
        
        Args:
            db_path: Path to store ChromaDB
            embedding_engine: Optional custom embedding engine
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Vector database will operate in demo mode.")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        if CHROMADB_AVAILABLE and chromadb:
            try:
                self.client = chromadb.PersistentClient(path=str(self.db_path))
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}. Operating in demo mode.")
                self.client = None
        else:
            self.client = None
        
        # Embedding engine
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        
        # Collections
        self.clinical_collection = None
        self.surgical_collection = None
        self.vqla_collection = None
        
        if self.client:
            self._initialize_collections()
        
        logger.info(f"Vector database initialized at {self.db_path}")
    
    def _initialize_collections(self):
        """Initialize or retrieve collections"""
        try:
            self.clinical_collection = self.client.get_or_create_collection(
                name=CHROMADB_COLLECTION_CLINICAL,
                metadata={"hnsw:space": "cosine"}
            )
            self.surgical_collection = self.client.get_or_create_collection(
                name=CHROMADB_COLLECTION_SURGICAL,
                metadata={"hnsw:space": "cosine"}
            )
            self.vqla_collection = self.client.get_or_create_collection(
                name=CHROMADB_COLLECTION_VQLA,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    def ingest_clinical_knowledge(self, documents: List[str], 
                                 metadata: Optional[List[Dict]] = None) -> int:
        """
        Ingest clinical knowledge documents
        
        Args:
            documents: List of clinical text documents
            metadata: Optional metadata for each document
        
        Returns:
            Number of documents ingested
        """
        if not self.clinical_collection:
            logger.warning("Vector DB not available. Skipping clinical ingestion.")
            return 0
        
        logger.info(f"Ingesting {len(documents)} clinical documents...")
        
        ids = []
        embeddings = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"clinical_{idx}"
            embedding = self.embedding_engine.embed_text(doc)
            
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(metadata[idx] if metadata else {"source": "clinical"})
        
        # Add to collection
        self.clinical_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Ingested {len(ids)} clinical documents")
        return len(ids)
    
    def ingest_surgical_reasoning(self, documents: List[str],
                                 metadata: Optional[List[Dict]] = None) -> int:
        """
        Ingest surgical reasoning and procedure documents
        
        Args:
            documents: List of surgical reasoning texts
            metadata: Optional metadata
        
        Returns:
            Number of documents ingested
        """
        logger.info(f"Ingesting {len(documents)} surgical documents...")
        
        ids = []
        embeddings = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"surgical_{idx}"
            embedding = self.embedding_engine.embed_text(doc)
            
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(metadata[idx] if metadata else {"source": "surgical"})
        
        self.surgical_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Ingested {len(ids)} surgical documents")
        return len(ids)
    
    def ingest_vqla_labels(self, labels: List[Dict], 
                           image_ids: List[str]) -> int:
        """
        Ingest VQLA surgical question-answer pairs and labels
        
        Args:
            labels: List of VQLA label dictionaries
            image_ids: Corresponding image identifiers
        
        Returns:
            Number of labels ingested
        """
        logger.info(f"Ingesting {len(labels)} VQLA labels...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for idx, (label_dict, img_id) in enumerate(zip(labels, image_ids)):
            # Convert label dict to text representation
            label_text = json.dumps(label_dict) if isinstance(label_dict, dict) else str(label_dict)
            
            doc_id = f"vqla_{img_id}_{idx}"
            embedding = self.embedding_engine.embed_text(label_text)
            
            ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append({
                "image_id": img_id,
                "source": "vqla",
                "label_type": label_dict.get("type", "unknown") if isinstance(label_dict, dict) else "unknown"
            })
            documents.append(label_text)
        
        self.vqla_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"Ingested {len(ids)} VQLA labels")
        return len(ids)
    
    def retrieve_clinical_knowledge(self, query: str, 
                                   top_k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieve relevant clinical documents
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            Tuple of (documents, distances, metadata)
        """
        if not self.clinical_collection:
            logger.warning("Vector DB not available. Returning empty results.")
            return [], [], []
        
        results = self.clinical_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadata = results['metadatas'][0] if results['metadatas'] else []
        
        return documents, distances, metadata
    
    def retrieve_surgical_reasoning(self, query: str,
                                   top_k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieve relevant surgical reasoning
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            Tuple of (documents, distances, metadata)
        """
        if not self.surgical_collection:
            logger.warning("Vector DB not available. Returning empty results.")
            return [], [], []
        
        results = self.surgical_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadata = results['metadatas'][0] if results['metadatas'] else []
        
        return documents, distances, metadata
    
    def retrieve_vqla_context(self, query: str,
                             top_k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Retrieve relevant VQLA labels and contexts
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            Tuple of (documents, distances, metadata)
        """
        if not self.vqla_collection:
            logger.warning("Vector DB not available. Returning empty results.")
            return [], [], []
        
        results = self.vqla_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadata = results['metadatas'][0] if results['metadatas'] else []
        
        return documents, distances, metadata
    
    def hybrid_retrieve(self, query: str, 
                       top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve from all collections for comprehensive context
        
        Args:
            query: Query text
            top_k: Results per collection
        
        Returns:
            Dictionary with results from each collection
        """
        return {
            'clinical': self.retrieve_clinical_knowledge(query, top_k),
            'surgical': self.retrieve_surgical_reasoning(query, top_k),
            'vqla': self.retrieve_vqla_context(query, top_k)
        }
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about collections"""
        return {
            'clinical': self.clinical_collection.count() if self.clinical_collection else 0,
            'surgical': self.surgical_collection.count() if self.surgical_collection else 0,
            'vqla': self.vqla_collection.count() if self.vqla_collection else 0,
            'status': 'chromadb_available' if CHROMADB_AVAILABLE else 'demo_mode'
        }


if __name__ == "__main__":
    # Demo usage
    vdb = VectorDatabase()
    
    # Sample clinical documents
    clinical_docs = [
        "Patient presents with elevated troponin levels indicating myocardial injury",
        "Anticoagulation therapy contraindicated due to active bleeding",
        "EF reduced to 35% consistent with dilated cardiomyopathy"
    ]
    
    # Ingest
    vdb.ingest_clinical_knowledge(clinical_docs)
    
    # Retrieve
    retrieved = vdb.retrieve_clinical_knowledge("cardiac risk assessment")
    print("Retrieved documents:", retrieved[0])
    print("Distances:", retrieved[1])
