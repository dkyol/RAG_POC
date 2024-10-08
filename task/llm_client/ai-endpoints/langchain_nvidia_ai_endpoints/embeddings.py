"""Embeddings Components Derived from NVEModel/Embeddings"""

import warnings
from typing import Any, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.pydantic_v1 import Field, validator

from langchain_nvidia_ai_endpoints._common import BaseNVIDIA
from langchain_nvidia_ai_endpoints.callbacks import usage_callback_var

from ._statics import MODEL_SPECS


class NVIDIAEmbeddings(BaseNVIDIA, Embeddings):
    """
    Client to NVIDIA embeddings models.

    Fields:
    - model: str, the name of the model to use
    - truncate: "NONE", "START", "END", truncate input text if it exceeds the model's
        maximum token length. Default is "NONE", which raises an error if an input is
        too long.
    """

    _default_model: str = "nvidia/embed-qa-4"
    _default_max_batch_size: int = 50
    model: str = Field(_default_model, description="Name of the model to invoke")
    truncate: Optional[Literal["NONE", "START", "END"]] = Field(
        description=(
            "Truncate input text from START or END if max token length is exceeded."
            " If 'NONE', exceeding inputs will trigger exception. Unset by default."
        ),
    )
    max_length: int = Field(2048, ge=1)
    max_batch_size: int = Field(default=_default_max_batch_size)
    model_type: Optional[Literal["passage", "query"]] = Field(
        description="The type of text to be embedded."
    )

    # indicate to user that max_length is deprecated when passed as an argument to
    # NVIDIAEmbeddings' constructor, e.g. NVIDIAEmbeddings(max_length=...). this
    # does not warning on assignment, e.g. embedder.max_length = ...
    # todo: fix _NVIDIAClient.validate_client and enable Config.validate_assignment
    @validator("max_length")
    def deprecated_max_length(cls, value: int) -> int:
        """Deprecate the max_length field."""
        warnings.warn(
            "The max_length field is deprecated. Use the 'truncate' instead.",
            DeprecationWarning,
        )
        return value

    # todo: fix _NVIDIAClient.validate_client and enable Config.validate_assignment
    @validator("model")
    def deprecated_nvolveqa_40k(cls, value: str) -> str:
        """Deprecate the nvolveqa_40k model."""
        if value == "nvolveqa_40k" or value == "playground_nvolveqa_40k":
            warnings.warn(
                "nvolveqa_40k is deprecated. Use 'nvidia/embed-qa-4' instead.",
                DeprecationWarning,
            )
        return value

    def _embed(
        self, texts: List[str], model_type: Literal["passage", "query"], **kwargs: Any
    ) -> List[List[float]]:
        """Embed a single text entry to either passage or query type"""
        # AI Foundation Model API -
        #  input: str | list[str]              -- <= 2048 characters, <= 50 inputs
        #  model: "query" | "passage"          -- type of input text to be embedded
        #  encoding_format: "float" | "base64"
        # API Catalog API -
        #  input: str | list[str]              -- char limit depends on model
        #  model: str                          -- model name, e.g. NV-Embed-QA
        #  encoding_format: "float" | "base64"
        #  input_type: "query" | "passage"
        #  user: str                           -- ignored
        #  truncate: "NONE" | "START" | "END"  -- default "NONE", error raised if
        #                                         an input is too long
        attr_kwargs = {
            "input": texts,
            "encoding_format": "float",
        }
        if self.truncate:
            attr_kwargs["truncate"] = self.truncate
        default_kwargs = self.default_kwargs(self.__class__.__name__)
        attr_kwargs = {k: v for k, v in attr_kwargs.items() if v is not None}
        payload = {**default_kwargs, **attr_kwargs, **kwargs}
        ## TODO: Remove aifm edge case
        if MODEL_SPECS.get(self.model, {}).get("api_type", None) == "aifm":
            payload["model"] = model_type
            payload.pop("truncate", "")
        else: 
            payload["input_type"] = model_type
        result = self.client.get_req_generation(self.model, payload=payload)
        data = result.get("data", result)
        if not isinstance(data, list):
            raise ValueError(f"Expected data with a list of embeddings. Got: {data}")
        embedding_list = [(res.get("embedding"), res.get("index")) for res in data]
        self._invoke_callback_vars(result)
        vectors = [x[0] for x in sorted(embedding_list, key=lambda x: x[1])]
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Input pathway for query embeddings."""
        return self._embed([text], model_type=self.model_type or "query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Input pathway for document embeddings."""
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise ValueError(f"`texts` must be a list of strings, given: {repr(texts)}")

        # From https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-40k/documentation
        # The input must not exceed the 2048 max input characters and inputs above 512
        # model tokens will be truncated. The input array must not exceed 50 input
        #  strings.
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            truncated = [
                text[: self.max_length] if len(text) > self.max_length else text
                for text in batch
            ]
            all_embeddings.extend(
                self._embed(truncated, model_type=self.model_type or "passage")
            )
        return all_embeddings

    def _invoke_callback_vars(self, response: dict) -> None:
        """Invoke the callback context variables if there are any."""
        callback_vars = [
            usage_callback_var.get(),
        ]
        llm_output = {**response, "model_name": self.model}
        result = LLMResult(generations=[[]], llm_output=llm_output)
        for cb_var in callback_vars:
            if cb_var:
                cb_var.on_llm_end(result)
