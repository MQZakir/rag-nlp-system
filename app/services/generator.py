"""
LLM generation layer.

Wraps LangChain to support three providers:
  - OpenAI (gpt-4o-mini, gpt-4o, etc.)
  - Anthropic (claude-3-haiku, claude-3-sonnet, etc.)
  - Ollama (any locally served model)

The RAG prompt is carefully structured to:
  1. Provide numbered source passages with their metadata.
  2. Instruct the model to cite sources by number.
  3. Encourage "I don't know" responses when context is insufficient,
     rather than hallucinating.

Streaming is supported via LangChain's `astream` interface, which yields
token chunks that the FastAPI route converts to SSE events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.logging import get_logger
from app.services.retriever import RetrievedChunk

log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. You will be given a question and \
a set of numbered source passages retrieved from a document corpus.

Instructions:
- Answer the question using only the information in the provided sources.
- Cite sources by their number in square brackets, e.g. [1] or [2, 3].
- If multiple sources support a claim, cite all relevant ones.
- If the sources do not contain enough information to answer the question, \
say "I don't have enough information in the provided context to answer this question" \
and briefly explain what is missing.
- Do not invent facts, statistics, or quotes that are not present in the sources.
- Be concise but complete. Prefer structured answers (bullet points, numbered lists) \
for multi-part questions.
"""

_HUMAN_TEMPLATE = """\
Sources:
{sources}

Question: {query}

Answer:"""


def _format_sources(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta_str = ", ".join(
            f"{k}: {v}"
            for k, v in chunk.metadata.items()
            if k not in ("doc_index", "start_index")
        )
        header = f"[{i}]" + (f" ({meta_str})" if meta_str else "")
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def _build_llm() -> BaseChatModel:
    provider = settings.llm_provider
    log.info("initialising LLM", provider=provider, model=settings.llm_model)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(  # type: ignore[call-arg]
            model=settings.llm_model,
            api_key=settings.llm_api_key,  # type: ignore[arg-type]
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


class GeneratorService:
    """Builds a LangChain prompt chain and exposes sync + async generation."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                ("human", _HUMAN_TEMPLATE),
            ]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Synchronous generation — used for non-streaming requests."""
        sources_text = _format_sources(chunks)
        log.debug("generating answer", query_len=len(query), sources=len(chunks))
        answer: str = self._chain.invoke({"query": query, "sources": sources_text})
        return answer

    async def agenerate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Async generation — avoids blocking the event loop."""
        sources_text = _format_sources(chunks)
        answer: str = await self._chain.ainvoke({"query": query, "sources": sources_text})
        return answer

    async def astream(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Async token streaming — yields partial string chunks."""
        sources_text = _format_sources(chunks)
        async for token in self._chain.astream({"query": query, "sources": sources_text}):
            yield token


@lru_cache(maxsize=1)
def get_generator() -> GeneratorService:
    return GeneratorService(llm=_build_llm())
