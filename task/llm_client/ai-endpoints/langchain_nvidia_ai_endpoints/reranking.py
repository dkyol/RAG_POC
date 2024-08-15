from __future__ import annotations

from typing import Any, Generator, List, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr

from langchain_nvidia_ai_endpoints._common import BaseNVIDIA
from langchain_nvidia_ai_endpoints._statics import Model


class Ranking(BaseModel):
    index: int
    logit: float


class NVIDIARerank(BaseNVIDIA, BaseDocumentCompressor):
    """
    LangChain Document Compressor that uses the NVIDIA NeMo Retriever Reranking API.
    """

    _default_model: str = "nvidia/rerank-qa-mistral-4b"
    top_n: int = Field(5, ge=0, description="The number of documents to return.")
    model: str = Field(_default_model, description="The model to use for reranking.")
    max_batch_size: int = Field(32, ge=1, description="The maximum batch size.")

    # todo: batching when len(documents) > endpoint's max batch size
    def _rank(self, documents: List[str], query: str, **kwargs: Any) -> List[Ranking]:
        top_n = kwargs.get("top_n") or self.top_n
        default_kwargs = self.default_kwargs({"client": self.__class__.__name__})
        payload = {
            "query": {"text": query},
            "passages": [{"text": passage} for passage in documents],
            **default_kwargs
        }
        result = self.client.get_req_generation(self.model, payload=payload)
        rankings = result.get("data", result).get("rankings")
        # todo: callback support
        return [Ranking(**ranking) for ranking in rankings[:top_n]]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any
    ) -> Sequence[Document]:
        """
        Compress documents using the NVIDIA NeMo Retriever Reranking microservice API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        top_n = kwargs.get("top_n") or self.top_n
        if len(documents) == 0 or top_n < 1:
            return []

        def batch(ls: list, size: int) -> Generator[List[Document], None, None]:
            for i in range(0, len(ls), size):
                yield ls[i : i + size]

        doc_list = list(documents)
        results = []
        for doc_batch in batch(doc_list, self.max_batch_size):
            rankings = self._rank(
                query=query, documents=[d.page_content for d in doc_batch]
            )
            for ranking in rankings:
                doc = doc_batch[ranking.index]
                doc.metadata["relevance_score"] = ranking.logit
                results.append(doc)

        # if we batched, we need to sort the results
        if len(doc_list) > self.max_batch_size:
            results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return results[:top_n]
