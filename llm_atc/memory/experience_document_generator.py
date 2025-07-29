# memory/experience_document_generator.py
"""
Experience Document Generator for LLM-ATC-HAL
Generates structured experience documents and creates 1024-dimensional embeddings
using Hugging Face's intfloat/e5-large-v2 model with local Chroma HNSW storage.
"""

import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class ExperienceDocument:
    """Structured experience document for storage and retrieval"""
    experience_id: str
    timestamp: float
    conflict_type: str  # 'convergent', 'parallel', 'crossing', 'overtaking'
    num_aircraft: int
    scenario_text: str  # Natural language description
    conflict_geometry_text: str  # Geometry description
    environmental_text: str  # Environmental conditions
    llm_decision_text: str  # LLM decision description
    baseline_decision_text: str  # Baseline decision description
    outcome_text: str  # Outcome description
    lessons_learned: str
    safety_margin: float
    icao_compliant: bool
    hallucination_detected: bool
    hallucination_types: list[str]
    metadata: dict[str, Any]


class ExperienceDocumentGenerator:
    """Generates structured experience documents from raw conflict data"""

    def __init__(self, persist_directory: str = "memory/chroma_experience_library"):
        """
        Initialize the experience document generator with E5-large-v2 embeddings

        Args:
            persist_directory: Directory for Chroma persistence
        """
        self.persist_directory = persist_directory
        self.logger = logging.getLogger(__name__)

        # Initialize sentence transformer model (E5-large-v2, ~1024-dim)
        try:
            self.embedding_model = SentenceTransformer("intfloat/e5-large-v2")
            self.embedding_dim = 1024
            self.logger.info("Loaded E5-large-v2 model successfully")
        except Exception:
            self.logger.exception("Failed to load E5-large-v2 model")
            raise

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create or get collection with local embeddings (no embedding function)
        self.collection_name = "atc_experiences_e5_large"
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
            )
            self.logger.info("Connected to existing collection: %s", self.collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=None,  # Use local embeddings
                metadata={
                    "hnsw:space": "cosine",  # HNSW with cosine similarity
                    "embedding_model": "intfloat/e5-large-v2",
                    "embedding_dim": self.embedding_dim,
                },
            )
            self.logger.info("Created new collection: %s", self.collection_name)

    def generate_experience(self,
                          conflict_desc: str,
                          commands_do: list[str],
                          commands_dont: list[str],
                          reasoning: str,
                          conflict_type: str,
                          num_ac: int,
                          **kwargs) -> dict:
        """
        Generate a structured experience document

        Args:
            conflict_desc: Description of the conflict situation
            commands_do: List of recommended commands/actions
            commands_dont: List of commands/actions to avoid
            reasoning: Reasoning behind the recommendations
            conflict_type: Type of conflict ('convergent', 'parallel', 'crossing', 'overtaking')
            num_ac: Number of aircraft involved
            **kwargs: Additional metadata fields

        Returns:
            Dict containing the experience document and metadata
        """

        try:
            experience_id = kwargs.get("experience_id", str(uuid.uuid4()))

            # Generate comprehensive text descriptions
            scenario_text = self._generate_scenario_text(conflict_desc, num_ac, conflict_type)
            llm_decision_text = self._generate_decision_text(commands_do, commands_dont, reasoning)

            # Create experience document
            experience_doc = ExperienceDocument(
                experience_id=experience_id,
                timestamp=time.time(),
                conflict_type=conflict_type,
                num_aircraft=num_ac,
                scenario_text=scenario_text,
                conflict_geometry_text=conflict_desc,
                environmental_text=kwargs.get("environmental_conditions", ""),
                llm_decision_text=llm_decision_text,
                baseline_decision_text=kwargs.get("baseline_decision", ""),
                outcome_text=kwargs.get("outcome", ""),
                lessons_learned=reasoning,
                safety_margin=kwargs.get("safety_margin", 0.0),
                icao_compliant=kwargs.get("icao_compliant", True),
                hallucination_detected=kwargs.get("hallucination_detected", False),
                hallucination_types=kwargs.get("hallucination_types", []),
                metadata={
                    "commands_do": commands_do,
                    "commands_dont": commands_dont,
                    "reasoning": reasoning,
                    "conflict_type": conflict_type,
                    "num_ac": num_ac,
                    **kwargs.get("additional_metadata", {}),
                },
            )

            return asdict(experience_doc)

        except Exception as e:
            self.logger.exception("Failed to generate experience document")
            # Return minimal document
            return {
                "experience_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "conflict_type": conflict_type,
                "num_aircraft": num_ac,
                "scenario_text": conflict_desc,
                "error": str(e),
            }

    def embed_and_store(self, exp_doc: dict) -> None:
        """
        Embed the experience document and store in Chroma

        Args:
            exp_doc: Experience document dictionary
        """
        try:
            # Create embedding from conflict description
            conflict_desc = exp_doc.get("conflict_geometry_text", exp_doc.get("scenario_text", ""))
            if not conflict_desc:
                self.logger.warning("No conflict description found for embedding")
                return

            # Generate embedding using E5-large-v2
            embedding = self.embedding_model.encode(
                conflict_desc,
                normalize_embeddings=True,
            )

            # Ensure embedding is the right shape and type
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            # Prepare metadata for filtering
            metadata = {
                "conflict_type": exp_doc.get("conflict_type", "unknown"),
                "num_ac": exp_doc.get("num_aircraft", 0),
                "timestamp": exp_doc.get("timestamp", time.time()),
                "experience_id": exp_doc.get("experience_id", str(uuid.uuid4())),
                "safety_margin": exp_doc.get("safety_margin", 0.0),
                "icao_compliant": exp_doc.get("icao_compliant", True),
            }

            # Store in Chroma
            self.collection.upsert(
                ids=[exp_doc["experience_id"]],
                embeddings=[embedding],
                documents=[conflict_desc],
                metadatas=[metadata],
            )

            self.logger.info("Successfully stored experience %s", exp_doc["experience_id"])

        except Exception:
            self.logger.exception("Failed to embed and store experience")
            raise

    def _generate_scenario_text(self, conflict_desc: str, num_ac: int, conflict_type: str) -> str:
        """Generate comprehensive scenario description"""
        return f"Conflict scenario involving {num_ac} aircraft in a {conflict_type} conflict situation. {conflict_desc}"

    def _generate_decision_text(self, commands_do: list[str], commands_dont: list[str], reasoning: str) -> str:
        """Generate decision description text"""
        do_text = "; ".join(commands_do) if commands_do else "No specific actions recommended"
        dont_text = "; ".join(commands_dont) if commands_dont else "No specific restrictions"
        return f"Recommended actions: {do_text}. Avoid: {dont_text}. Reasoning: {reasoning}"

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the stored experiences"""
        try:
            count = self.collection.count()
            return {
                "total_experiences": count,
                "collection_name": self.collection_name,
                "embedding_model": "intfloat/e5-large-v2",
                "embedding_dim": self.embedding_dim,
            }
        except Exception as e:
            self.logger.exception("Failed to get collection stats")
            return {"error": str(e)}
