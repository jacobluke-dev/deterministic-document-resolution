from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")


@dataclass(frozen=True, slots=True)
class _DemoChunk:
    chunk_id: str
    document_id: str
    document_name: str
    ordinal: int
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class _RetrievedChunk:
    chunk: _DemoChunk
    score: float


class _GroundingStage:
    async def ground_documents(self, documents):
        return tuple(documents)


class _Chunker:
    def chunk_documents(self, documents):
        document = documents[0]
        return (
            _DemoChunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                document_name=document.name,
                ordinal=0,
                start_offset=0,
                end_offset=len(document.text),
                text=document.text,
            ),
        )


class _ChunkerStub:
    def chunk_documents(self, documents):
        document = documents[0]
        return (
            _DemoChunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                document_name=document.name,
                ordinal=0,
                start_offset=0,
                end_offset=len(document.text),
                text=document.text,
            ),
        )


@pytest.fixture
def grounding_stage() -> _GroundingStage:
    return _GroundingStage()


@pytest.fixture
def chunker() -> _Chunker:
    return _Chunker()


@pytest.fixture
def chunk_stub() -> _ChunkerStub:
    return _ChunkerStub()


@pytest.fixture
def demo_chunk() -> Callable[..., _DemoChunk]:
    def build(**kwargs: Any) -> _DemoChunk:
        return _DemoChunk(**kwargs)

    return build


@pytest.fixture
def retrieved_chunk() -> Callable[..., _RetrievedChunk]:
    def build(**kwargs: Any) -> _RetrievedChunk:
        return _RetrievedChunk(**kwargs)

    return build
