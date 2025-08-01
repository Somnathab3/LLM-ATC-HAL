# memory/replay_store.py
"""
Vector-Based Experience Replay Store for LLM-ATC-HAL
Uses 1024-dimensional intfloat/e5-large-v2 embeddings with local Chroma HNSW storage
and metadata filtering for efficient experience retrieval.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

# Handle potential sentence transformers issue gracefully
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, SyntaxError) as e:
    logging.warning("SentenceTransformers not available: %s", e)
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class ConflictExperience:
    """Container for conflict experience data"""

    conflict_id: str = ""
    conflict_description: str = ""
    resolution_commands: list[str] = None
    safety_score: float = 0.0
    conflict_type: str = ""
    num_aircraft: int = 0
    reasoning: str = ""
    outcome: str = ""
    metadata: dict[str, Any] = None
    experience_id: str = ""  # Add experience_id field with default
    timestamp: float = 0.0  # Add timestamp field with default
    scenario_context: dict[str, Any] = None  # Add scenario_context field
    conflict_geometry: dict[str, Any] = None  # Add conflict_geometry field
    environmental_conditions: dict[str, Any] = (
        None  # Add environmental_conditions field
    )
    llm_decision: dict[str, Any] = None  # Add llm_decision field
    baseline_decision: dict[str, Any] = None  # Add baseline_decision field
    actual_outcome: dict[str, Any] = None  # Add actual_outcome field
    safety_metrics: dict[str, Any] = None  # Add safety_metrics field
    hallucination_detected: bool = False  # Add hallucination_detected field
    hallucination_types: list[str] = None  # Add hallucination_types field
    controller_override: Any = None  # Add controller_override field
    lessons_learned: str = ""  # Add lessons_learned field

    def __post_init__(self):
        if self.resolution_commands is None:
            self.resolution_commands = []
        if self.metadata is None:
            self.metadata = {}
        if self.scenario_context is None:
            self.scenario_context = {}
        if self.conflict_geometry is None:
            self.conflict_geometry = {}
        if self.environmental_conditions is None:
            self.environmental_conditions = {}
        if self.llm_decision is None:
            self.llm_decision = {}
        if self.baseline_decision is None:
            self.baseline_decision = {}
        if self.actual_outcome is None:
            self.actual_outcome = {}
        if self.safety_metrics is None:
            self.safety_metrics = {}
        if self.hallucination_types is None:
            self.hallucination_types = []


@dataclass
class SimilarityResult:
    """Container for similarity search results"""

    experience: ConflictExperience
    similarity_score: float
    metadata: dict[str, Any]


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
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("intfloat/e5-large-v2")
                self.embedding_dim = 1024
                self.logger.info(
                    "Initialized SentenceTransformer model: intfloat/e5-large-v2"
                )
            except Exception as e:
                self.logger.warning("Failed to initialize SentenceTransformer: %s", e)
                self.embedding_model = None
                self.embedding_dim = 1024
        else:
            self.logger.warning(
                "SentenceTransformers not available, using fallback embedding"
            )
            self.embedding_model = None
            self.embedding_dim = 1024

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
            self.logger.info(
                "Connected to existing collection: %s", self.collection_name
            )
        except Exception:
            self.logger.warning(
                "Collection %s not found, creating new one", self.collection_name
            )
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # Use local embeddings
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": "intfloat/e5-large-v2",
                    "embedding_dim": self.embedding_dim,
                },
            )

    def store_experience(self, experience: ConflictExperience) -> str:
        """
        Store a conflict experience in the vector store

        Args:
            experience: ConflictExperience object to store

        Returns:
            str: Experience ID if successful, empty string if failed
        """
        try:
            import uuid

            # Generate unique experience ID
            exp_id = f"exp_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Create text description for embedding
            conflict_desc = f"{experience.conflict_description} {experience.reasoning}"

            # Generate embedding
            if self.embedding_model:
                try:
                    embedding = self.embedding_model.encode(
                        conflict_desc,
                        normalize_embeddings=True,
                    )
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                except Exception as e:
                    self.logger.warning("Failed to generate embedding: %s", e)
                    # Create fallback random embedding
                    embedding = [0.0] * self.embedding_dim
            else:
                # Create fallback random embedding
                embedding = [0.0] * self.embedding_dim

            # Prepare metadata
            metadata = {
                "experience_id": exp_id,
                "conflict_id": experience.conflict_id,
                "conflict_type": experience.conflict_type,
                "num_aircraft": experience.num_aircraft,
                "safety_score": experience.safety_score,
                "outcome": experience.outcome,
                "timestamp": experience.timestamp,
                "hallucination_detected": experience.hallucination_detected,
            }

            # Add metadata from experience.metadata if it exists
            if experience.metadata:
                metadata.update(experience.metadata)

            # Store in collection
            self.collection.add(
                ids=[exp_id],
                embeddings=[embedding],
                documents=[conflict_desc],
                metadatas=[metadata],
            )

            self.logger.info("Stored experience %s", exp_id)
            return exp_id

        except Exception:
            self.logger.exception("Failed to store experience")
            return ""

    def retrieve_experience(
        self,
        conflict_desc: str,
        conflict_type: str,
        num_ac: int,
        k: int = 5,
    ) -> list[dict]:
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
            # Step 1: Metadata filtering using simple where clause
            # Try simple format first
            try:
                filtered_results = self.collection.get(
                    where={"conflict_type": conflict_type, "num_ac": num_ac},
                )
            except Exception:
                # If that fails, try $eq format
                try:
                    filtered_results = self.collection.get(
                        where={
                            "$and": [
                                {"conflict_type": {"$eq": conflict_type}},
                                {"num_ac": {"$eq": num_ac}},
                            ],
                        },
                    )
                except Exception:
                    # If both fail, skip metadata filtering
                    self.logger.warning(
                        "Metadata filtering failed, retrieving all documents"
                    )
                    filtered_results = self.collection.get()

            if not filtered_results["ids"]:
                self.logger.info(
                    "No experiences found for conflict_type=%s, num_ac=%s",
                    conflict_type,
                    num_ac,
                )
                return []

            self.logger.info(
                "Found %d experiences matching metadata filters",
                len(filtered_results["ids"]),
            )

            # Step 2: Vector search on filtered results
            query_embedding = self.embedding_model.encode(
                conflict_desc,
                normalize_embeddings=True,
            )

            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Query with embedding, try simple format first
            try:
                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k, len(filtered_results["ids"])),
                    where={"conflict_type": conflict_type, "num_ac": num_ac},
                )
            except Exception:
                # If that fails, try without where clause
                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                )

            # Format results
            experiences = []
            if search_results["ids"] and search_results["ids"][0]:
                for i, exp_id in enumerate(search_results["ids"][0]):
                    distance = (
                        search_results["distances"][0][i]
                        if search_results["distances"]
                        else 1.0
                    )
                    similarity = 1.0 - distance  # Convert distance to similarity

                    metadata = (
                        search_results["metadatas"][0][i]
                        if search_results["metadatas"]
                        else {}
                    )
                    document = (
                        search_results["documents"][0][i]
                        if search_results["documents"]
                        else ""
                    )

                    # Filter by metadata if database query didn't work
                    if (
                        metadata.get("conflict_type") == conflict_type
                        and metadata.get("num_ac") == num_ac
                    ):
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

    def get_all_experiences(
        self,
        conflict_type: Optional[str] = None,
        num_ac: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
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
            # Try simple filtering first
            if conflict_type and num_ac is not None:
                try:
                    results = self.collection.get(
                        where={"conflict_type": conflict_type, "num_ac": num_ac},
                        limit=limit,
                    )
                except Exception:
                    # If database filtering fails, get all and filter manually
                    results = self.collection.get(limit=limit)
            elif conflict_type:
                try:
                    results = self.collection.get(
                        where={"conflict_type": conflict_type},
                        limit=limit,
                    )
                except Exception:
                    results = self.collection.get(limit=limit)
            elif num_ac is not None:
                try:
                    results = self.collection.get(
                        where={"num_ac": num_ac},
                        limit=limit,
                    )
                except Exception:
                    results = self.collection.get(limit=limit)
            else:
                results = self.collection.get(limit=limit)

            experiences = []
            if results["ids"]:
                for i, exp_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    document = results["documents"][i] if results["documents"] else ""

                    # Manual filtering if database filtering failed
                    if conflict_type and metadata.get("conflict_type") != conflict_type:
                        continue
                    if num_ac is not None and metadata.get("num_ac") != num_ac:
                        continue

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

            # Get breakdown by conflict type - use manual counting if needed
            conflict_types = {}
            all_experiences = self.get_all_experiences()

            for exp in all_experiences:
                ct = exp["metadata"].get("conflict_type", "unknown")
                conflict_types[ct] = conflict_types.get(ct, 0) + 1

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
