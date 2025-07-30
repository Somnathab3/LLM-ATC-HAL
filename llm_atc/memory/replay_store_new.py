# memory/replay_store.py
"""
Vector-Based Experience Replay Store for LLM-ATC-HAL
Uses 1024-dimensional intfloat/e5-large-v2 embeddings with local Chroma HNSW storage
and metadata filtering for efficient experience retrieval.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedExperience:
    """Container for retrieved experience with similarity score"""
    experience_id: str
    experience_data: dict[str, Any]
    similarity_score: float
    metadata: dict[str, Any]


class VectorReplayStore:
    """
    Vector-based experience replay store using 1024-dim intfloat/e5-large-v2
    with local Chroma HNSW and metadata filtering
    """

    def __init__(self, storage_dir: str = "memory/chroma_experience_library") -> None:
        """
        Initialize the replay store

        Args:
            storage_dir: Directory for Chroma persistence
        """
        self.storage_dir = storage_dir
        self.logger = logging.getLogger(__name__)

        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)

        # Initialize sentence transformer model (same as generator)
        try:
            self.embedding_model = SentenceTransformer("intfloat/e5-large-v2")
            self.embedding_dim = 1024
            self.logger.info("Loaded E5-large-v2 model for retrieval")
        except Exception:
            self.logger.exception("Failed to load E5-large-v2 model")
            raise

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=storage_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get existing collection
        self.collection_name = "atc_experiences_e5_large"
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
            )
            self.logger.info("Connected to existing collection: %s", self.collection_name)
        except Exception:
            self.logger.warning("Collection %s not found, creating new one", self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # Use local embeddings
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": "intfloat/e5-large-v2",
                    "embedding_dim": self.embedding_dim,
                },
            )

    def retrieve_experience(self,
                          conflict_desc: str,
                          conflict_type: str,
                          num_ac: int,
                          k: int = 5) -> list[dict]:
        """
        Retrieve similar experiences using metadata filtering + vector search

        Args:
            conflict_desc: Description of the conflict to search for
            conflict_type: Type of conflict to filter by
            num_ac: Number of aircraft to filter by
            k: Number of results to return

        Returns:
            List of experience documents in score-ascending order
        """
        try:
            # Step 1: Metadata filtering
            filtered_results = self.collection.get(
                where={
                    "conflict_type": conflict_type,
                    "num_ac": num_ac,
                },
            )

            if not filtered_results["ids"]:
                self.logger.info("No experiences found for conflict_type=%s, num_ac=%s", conflict_type, num_ac)
                return []

            self.logger.info("Found %d experiences matching metadata filters", len(filtered_results["ids"]))

            # Step 2: Vector search on filtered results
            query_embedding = self.embedding_model.encode(
                conflict_desc,
                normalize_embeddings=True,
            )

            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Query with embedding
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, len(filtered_results["ids"])),
                where={
                    "conflict_type": conflict_type,
                    "num_ac": num_ac,
                },
            )

            # Format results
            experiences = []
            if search_results["ids"] and search_results["ids"][0]:
                for i, exp_id in enumerate(search_results["ids"][0]):
                    distance = search_results["distances"][0][i] if search_results["distances"] else 1.0
                    similarity = 1.0 - distance  # Convert distance to similarity

                    metadata = search_results["metadatas"][0][i] if search_results["metadatas"] else {}
                    document = search_results["documents"][0][i] if search_results["documents"] else ""

                    experience_data = {
                        "experience_id": exp_id,
                        "conflict_desc": document,
                        "metadata": metadata,
                        "similarity_score": similarity,
                    }

                    experiences.append(experience_data)

            # Sort by similarity score (ascending distance = descending similarity)
            experiences.sort(key=lambda x: x["similarity_score"], reverse=True)

            self.logger.info("Retrieved %d similar experiences", len(experiences))
            return experiences

        except Exception:
            self.logger.exception("Failed to retrieve experiences")
            return []

    def get_all_experiences(self,
                           conflict_type: Optional[str] = None,
                           num_ac: Optional[int] = None,
                           limit: Optional[int] = None) -> list[dict]:
        """
        Get all experiences, optionally filtered by metadata

        Args:
            conflict_type: Optional conflict type filter
            num_ac: Optional number of aircraft filter
            limit: Optional limit on number of results

        Returns:
            List of experience documents
        """
        try:
            where_clause = {}
            if conflict_type:
                where_clause["conflict_type"] = conflict_type
            if num_ac is not None:
                where_clause["num_ac"] = num_ac

            results = self.collection.get(
                where=where_clause if where_clause else None,
                limit=limit,
            )

            experiences = []
            if results["ids"]:
                for i, exp_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    document = results["documents"][i] if results["documents"] else ""

                    experience_data = {
                        "experience_id": exp_id,
                        "conflict_desc": document,
                        "metadata": metadata,
                    }
                    experiences.append(experience_data)

            return experiences

        except Exception:
            self.logger.exception("Failed to get all experiences")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored experiences"""
        try:
            total_count = self.collection.count()

            # Get breakdown by conflict type
            conflict_types = {}
            for ct in ["convergent", "parallel", "crossing", "overtaking"]:
                count = len(self.collection.get(where={"conflict_type": ct})["ids"])
                if count > 0:
                    conflict_types[ct] = count

            return {
                "total_experiences": total_count,
                "by_conflict_type": conflict_types,
                "collection_name": self.collection_name,
                "embedding_model": "intfloat/e5-large-v2",
                "embedding_dim": self.embedding_dim,
                "storage_dir": self.storage_dir,
            }

        except Exception as e:
            self.logger.exception("Failed to get stats")
            return {"error": str(e)}

    def delete_experience(self, experience_id: str) -> bool:
        """Delete an experience by ID"""
        try:
            self.collection.delete(ids=[experience_id])
            self.logger.info("Deleted experience %s", experience_id)
            return True
        except Exception:
            self.logger.exception("Failed to delete experience %s", experience_id)
            return False

    def clear_all(self) -> bool:
        """Clear all experiences from the store"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": "intfloat/e5-large-v2",
                    "embedding_dim": self.embedding_dim,
                },
            )
            self.logger.info("Cleared all experiences")
            return True
        except Exception:
            self.logger.exception("Failed to clear all experiences")
            return False
