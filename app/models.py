"""Pydantic schemas for the whole system.

Every generated claim is labeled with a content type and a confidence value.
Structured output is not assumed: these schemas are used for validation AFTER
plain-text model output has been parsed.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    """How a piece of content should be treated."""

    OBSERVATION = "observation"
    MEMORY = "memory"
    USER_CLAIM = "user_claim"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    SPECULATION = "speculation"


class EventType(str, Enum):
    OBSERVATION = "observation"
    USER_STATEMENT = "user_statement"
    DECISION = "decision"
    ACTION = "action"
    OUTCOME = "outcome"
    REFLECTION = "reflection"
    HYPOTHESIS = "hypothesis"
    CONTRADICTION = "contradiction"
    SUMMARY = "summary"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    OBSERVATION = "observation"
    GOAL = "goal"
    QUESTION = "question"
    REJECTED_HYPOTHESIS = "rejected_hypothesis"
    APPROVED_HYPOTHESIS = "approved_hypothesis"
    SUMMARY = "summary"


class BeliefType(str, Enum):
    OBSERVATION = "observation"
    REMEMBERED_FACT = "remembered_fact"
    USER_CLAIM = "user_claim"
    INFERENCE = "inference"
    PREDICTION = "prediction"
    SPECULATION = "speculation"


class Event(BaseModel):
    """An immutable record of something that happened."""

    event_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    session_id: str = "default"
    event_type: EventType = EventType.OBSERVATION
    content: str
    source: str = "user"
    related_task: str = ""
    importance: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def dict_for_db(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "content": self.content,
            "source": self.source,
            "related_task": self.related_task,
            "importance": self.importance,
            "confidence": self.confidence,
            "metadata": json_dumps(self.metadata),
            "timestamp": self.timestamp,
        }


class Memory(BaseModel):
    """A stored memory with full provenance."""

    id: Optional[str] = None
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    source: str = "user"
    timestamp: float = Field(default_factory=time.time)
    confidence: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    importance: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    expires_at: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    related_task: str = ""
    event_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def dict_for_db(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "importance": self.importance,
            "tags": ",".join(self.tags),
            "expires_at": self.expires_at,
            "verification_status": self.verification_status.value,
            "related_task": self.related_task,
            "event_id": self.event_id,
            "metadata": json_dumps(self.metadata),
        }


class EvidenceRef(BaseModel):
    memory_id: str
    source: str = ""
    kind: ContentType = ContentType.OBSERVATION


class Belief(BaseModel):
    """A belief with evidence, confidence, and provenance."""

    id: Optional[str] = None
    content: str
    belief_type: BeliefType = BeliefType.INFERENCE
    confidence: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    evidence: List[EvidenceRef] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    expires_at: Optional[float] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        return v


class ContradictionEntry(BaseModel):
    belief_a_id: str
    belief_b_id: str
    description: str = Field(min_length=1)
    created_at: float = Field(default_factory=time.time)


class HypothesisItem(BaseModel):
    """A single output of the controlled wandering process."""

    text: str
    category: str = "connection"
    supporting_memory_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default_factory=lambda: 0.3, ge=0.0, le=1.0)
    novelty: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    relevance: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    testability: float = Field(default_factory=lambda: 0.5, ge=0.0, le=1.0)
    is_speculative: bool = True
    suggested_experiment: str = ""
    content_type: ContentType = ContentType.SPECULATION

    def dict_for_db(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category,
            "supporting_memory_ids": ",".join(self.supporting_memory_ids),
            "confidence": self.confidence,
            "novelty": self.novelty,
            "relevance": self.relevance,
            "testability": self.testability,
            "is_speculative": int(self.is_speculative),
            "suggested_experiment": self.suggested_experiment,
        }


class EvaluationScores(BaseModel):
    relevance: float = 0.0
    novelty: float = 0.0
    internal_consistency: float = 0.0
    evidence_quality: float = 0.0
    testability: float = 0.0
    contamination_risk: float = 0.0


class EvaluatorDecision(str, Enum):
    REJECT = "reject"
    ARCHIVE_AS_SPECULATION = "archive_as_speculation"
    SUGGEST_TO_ACTIVE_AGENT = "suggest_to_active_agent"
    PROMOTE_AFTER_VERIFICATION = "promote_after_verification"


class EvaluatorReview(BaseModel):
    hypothesis: HypothesisItem
    decision: EvaluatorDecision
    scores: EvaluationScores
    reason: str = ""
    rejected_flags: List[str] = Field(default_factory=list)


class ReflectionReport(BaseModel):
    session_summary: str = ""
    successful_decisions: List[str] = Field(default_factory=list)
    failed_predictions: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    candidate_memories: List[str] = Field(default_factory=list)
    candidate_goals: List[str] = Field(default_factory=list)
    possible_improvements: List[str] = Field(default_factory=list)
    confidence: float = Field(default_factory=lambda: 0.4, ge=0.0, le=1.0)

    def nonempty_fields(self) -> Dict[str, List[str]]:
        return {k: v for k, v in self.model_dump().items()
                if isinstance(v, list) and v}


class ActiveAnswer(BaseModel):
    """Structured metadata wrapper around an active-agent reply."""

    text: str
    cited_memory_ids: List[str] = Field(default_factory=list)
    facts_used: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    uncertainty: float = Field(default_factory=lambda: 0.3, ge=0.0, le=1.0)
    content_type: ContentType = ContentType.INFERENCE
    hypotheses_used: List[str] = Field(default_factory=list)
    raw_model_calls: int = 0
    tokens: Dict[str, int] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    category: str
    agent: str
    memory_strategy: str = ""
    seed: int = 0
    success: bool = False
    rubric: Dict[str, float] = Field(default_factory=dict)
    retrieval_precision: float = 0.0
    unsupported_claim_rate: float = 0.0
    contradiction_rate: float = 0.0
    hypothesis_usefulness: Optional[float] = None
    hypothesis_novelty: Optional[float] = None
    latency_ms: float = 0.0
    tokens: Dict[str, int] = Field(default_factory=dict)
    model_calls: int = 0
    answer_preview: str = ""
    errors: List[str] = Field(default_factory=list)
    error_analysis: List[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    experiment_id: str
    agent: str
    memory_strategy: str = ""
    sample_seed: Optional[int] = None
    tasks: List[TaskResult] = Field(default_factory=list)
    aggregate: Dict[str, float] = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def accuracy(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.success for t in self.tasks) / len(self.tasks)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)