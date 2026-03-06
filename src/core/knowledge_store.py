import time
import logging
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
from core.config import QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEYS

logger = logging.getLogger(__name__)

class TreclKnowledgeStore:
    """
    Central VectorDB wrapper for Trecl's Agentic RAG architecture.
    Uses Qdrant Cloud for storage and filtering, and Gemini for embeddings.
    
    Migrated to the google-genai SDK (google.genai.Client) as of March 2026.
    See: https://ai.google.dev/gemini-api/docs/migrate
    """
    
    COLLECTION_NAME = "trecl_knowledge"
    EMBEDDING_MODEL = "gemini-embedding-001"
    VECTOR_SIZE = 768  # Gemini embedding size (configured for storage efficiency)
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 100
    
    def __init__(self):
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in the environment.")
        if not GEMINI_API_KEYS:
            raise ValueError("GEMINI_API_KEYS must be set in the environment.")
        
        # Initialize Qdrant Client
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        
        self._ensure_collection_exists()

    def _get_gemini_client(self) -> genai.Client:
        """Creates a new google.genai Client with the next available API key."""
        return genai.Client(api_key=GEMINI_API_KEYS.get_next_key())

    def _ensure_collection_exists(self):
        """Idempotently creates the collection and payload indexes if they don't exist."""
        collections_response = self.client.get_collections()
        collection_names = [c.name for c in collections_response.collections]
        
        if self.COLLECTION_NAME not in collection_names:
            logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="company_name",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="source_type",
                field_schema="keyword"
            )
            self.client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name="timestamp_epoch",
                field_schema="integer"
            )

    def ingest(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Chunks text, embeds with task_type=RETRIEVAL_DOCUMENT, and stores in Qdrant.
        """
        if not texts:
            return

        if len(texts) != len(metadatas):
            raise ValueError("Lengths of texts and metadatas must match.")

        all_chunks = []
        all_metadatas = []
        
        for text, meta in zip(texts, metadatas):
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                # Attach chunk text to metadata for retrieval later
                chunk_meta = meta.copy()
                chunk_meta["content"] = chunk
                all_metadatas.append(chunk_meta)

        if not all_chunks:
            return

        logger.info(f"Embedding {len(all_chunks)} chunks into VectorDB...")
        
        # Use new google.genai Client for embeddings
        gemini = self._get_gemini_client()
        response = gemini.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=all_chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.VECTOR_SIZE
            )
        )
        
        embeddings = [e.values for e in response.embeddings]

        points = [
            PointStruct(
                id=self._generate_id_from_hash(chunk + str(meta.get("timestamp_epoch", ""))),
                vector=embedding,
                payload=meta
            )
            for chunk, embedding, meta in zip(all_chunks, embeddings, all_metadatas)
        ]

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )

    def search(
        self, 
        query: str, 
        company_name: str, 
        source_filter: Optional[str] = None, 
        max_age_days: int = 540,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Embeds query with task_type=RETRIEVAL_QUERY and searches Qdrant with filters.
        """
        # Embed Query using new google.genai Client
        gemini = self._get_gemini_client()
        query_response = gemini.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.VECTOR_SIZE
            )
        )
        
        query_vector = query_response.embeddings[0].values

        # Construct filters
        must_conditions = [
            FieldCondition(
                key="company_name",
                match=MatchValue(value=company_name)
            )
        ]
        
        if source_filter:
            must_conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value=source_filter)
                )
            )
            
        if max_age_days > 0:
            current_epoch = int(time.time())
            cutoff_epoch = current_epoch - (max_age_days * 24 * 60 * 60)
            must_conditions.append(
                FieldCondition(
                    key="timestamp_epoch",
                    range=Range(gte=cutoff_epoch)
                )
            )

        search_filter = Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k
        )

        return [res.payload for res in results.points]

    def clear(self, company_name: str):
        """Clears all points associated with a specific company."""
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="company_name",
                        match=MatchValue(value=company_name)
                    )
                ]
            )
        )
        
    def _generate_id_from_hash(self, content: str) -> str:
        """Utility to generate a stable UUID from content hash."""
        import hashlib
        import uuid
        hash_md5 = hashlib.md5(content.encode('utf-8')).hexdigest()
        return str(uuid.UUID(hash_md5))
