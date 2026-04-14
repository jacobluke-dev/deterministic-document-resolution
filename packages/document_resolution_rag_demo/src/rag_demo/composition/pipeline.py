from __future__ import annotations

from openai import OpenAI
from public_api.core.services.resolve_service import ResolveService

from rag_demo.agentic.orchestrator import SingleAgentEvidenceOrchestrator
from rag_demo.agentic.prompted_reviewer import PromptedGroundingReviewer
from rag_demo.agentic.reviewer_model import OpenAIReviewerModel
from rag_demo.answering import DemoAnswerGenerator
from rag_demo.chunking import FixedWindowChunker
from rag_demo.composition.embedder import build_openai_embedder
from rag_demo.pipelines.baseline import BaselineRagPipeline
from rag_demo.pipelines.grounded import GroundedRagPipeline, ResolveBackedGroundingStage
from rag_demo.retrieval import FaissVectorStore
from rag_demo.settings import RagDemoSettings, get_rag_demo_settings


def build_baseline_pipeline(settings: RagDemoSettings | None = None) -> BaselineRagPipeline:
    """Build the baseline RAG pipeline from package settings.

    The baseline pipeline uses deterministic fixed-window chunking, OpenAI
    embeddings, FAISS-backed retrieval, and a simple baseline answer generator
    that answers only from retrieved context without deterministic grounding.

    Args:
        settings: RAG demo settings providing chunking and embedding
            configuration.

    Returns:
        A configured ``BaselineRagPipeline`` instance ready to index documents
        and answer questions.
    """
    settings = settings or get_rag_demo_settings()
    embedder = build_openai_embedder(settings)

    baseline_client = OpenAI(api_key=settings.openai_api_key)
    baseline_model = OpenAIReviewerModel(
        client=baseline_client,
        model=settings.reviewer_model,
    )

    return BaselineRagPipeline(
        chunker=FixedWindowChunker(
            chunk_size=settings.baseline_chunk_size,
            chunk_overlap=settings.baseline_chunk_overlap,
        ),
        vector_store=FaissVectorStore(embedder=embedder),
        answer_generator=DemoAnswerGenerator(
            model_complete=baseline_model.complete,
        ),
    )


def build_grounded_pipeline(
    *,
    resolve_service: ResolveService,
    settings: RagDemoSettings | None = None,
) -> GroundedRagPipeline:
    """Build the grounded RAG pipeline from package settings.

    The grounded pipeline performs deterministic grounding before retrieval and
    then applies a bounded single-agent evidence orchestrator after retrieval
    and before final answer generation.

    Args:
        resolve_service: Resolve service instance used to ground documents.
        settings: RAG demo settings providing chunking and embedding
            configuration.

    Returns:
        A configured ``GroundedRagPipeline`` instance ready to index documents
        and answer questions.
    """
    settings = settings or get_rag_demo_settings()
    embedder = build_openai_embedder(settings)

    reviewer_model = OpenAIReviewerModel(
        client=OpenAI(api_key=settings.openai_api_key),
        model=settings.reviewer_model,
    )

    model_complete = reviewer_model.complete

    return GroundedRagPipeline(
        chunker=FixedWindowChunker(
            chunk_size=settings.grounded_chunk_size,
            chunk_overlap=settings.grounded_chunk_overlap,
        ),
        vector_store=FaissVectorStore(embedder=embedder),
        grounding_stage=ResolveBackedGroundingStage(resolve_service=resolve_service),
        evidence_orchestrator=SingleAgentEvidenceOrchestrator(
            reviewer=PromptedGroundingReviewer(model_complete=model_complete),
        ),
    )
