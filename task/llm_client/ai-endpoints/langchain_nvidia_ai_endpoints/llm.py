from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import Field

from langchain_nvidia_ai_endpoints._common import BaseNVIDIA

_CallbackManager = Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]


"""
### Code Generation

These models accept the same arguments and input structure as regular chat models, but
they tend to perform better on code-genreation and structured code tasks. An example
of this is `llama2_code_70b`.

```
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert coding AI. Respond only in valid python;
            no narration whatsoever.",
        ),
        ("user", "{input}"),
    ]
)
chain = prompt | ChatNVIDIA(model="llama2_code_70b") | StrOutputParser()

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    print(txt, end="")
```

In addition, the [**StarCoder2**]
(https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/starcoder2-15b)
model also supports code generation, but subscribes to a regular completion API. For
this, you should use the LLM-style `NVIDIA` class:

```
from langchain_nvidia_ai_endpoints import NVIDIA

starcoder = NVIDIA(model="starcoder2_15b", stop=["```"])

# print(chain.invoke("Here is my implementation of fizzbuzz:\n```python\n", stop="```"))
for txt in starcoder.stream("Here is my implementation of fizzbuzz:\n```python\n"):
    print(txt, end="")
```
"""


class NVIDIA(BaseNVIDIA, LLM):
    """NVIDIA chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_ai_endpoints import ChatNVIDIA


            model = NVIDIA(model="starcoder2_15b")
            response = model.invoke("Here is my fizzbuzz code:\n```python\n")
    """

    _default_model: str = "starcoder2_15b"
    infer_endpoint: str = Field("{base_url}/completions")
    model: str = Field(_default_model, description="Name of the model to invoke")
    temperature: Optional[float] = Field(description="Sampling temperature in [0, 1]")
    max_tokens: Optional[int] = Field(description="Maximum # of tokens to generate")
    top_p: Optional[float] = Field(description="Top-p for distribution sampling")
    frequency_penalty: Optional[float] = Field(description="Frequency penalty")
    presence_penalty: Optional[float] = Field(description="Presence penalty")
    seed: Optional[int] = Field(description="The seed for deterministic results")
    bad: Optional[Sequence[str]] = Field(description="Bad words to avoid (cased)")
    stop: Optional[Sequence[str]] = Field(description="Stop words (cased)")
    labels: Optional[Dict[str, float]] = Field(description="Steering parameters")
    streaming: bool = Field(True)

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "nvidia-ai-playground"

    def _call(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Invoke on a single list of chat messages."""
        response = self.get_generation(prompt=prompt, **kwargs)
        output = self.postprocess(response)
        return output

    def _stream(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Allows streaming to model!"""
        for response in self.get_stream(prompt=prompt, **kwargs):
            self._set_callback_out(response, run_manager)
            chunk = self.postprocess(response, return_chunk=True)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        prompt: str,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for response in self.get_astream(prompt=prompt, **kwargs):
            self._set_callback_out(response, run_manager)
            chunk = self.postprocess(response, return_chunk=True)
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _set_callback_out(
        self,
        result: dict,
        run_manager: Optional[_CallbackManager],
    ) -> None:
        result.update({"model_name": self.model})
        if run_manager:
            for cb in run_manager.handlers:
                if hasattr(cb, "llm_output"):
                    cb.llm_output = result

    def postprocess(self, msg: dict, return_chunk=False) -> str:
        generation = (
            msg.get("cumulative") 
            or msg.get("choices") 
            or [{}]
        )[0].get("content", "")
        return GenerationChunk(text=generation) if return_chunk else generation

    ######################################################################################
    ## Core client-side interfaces

    def get_generation(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        """Call to client generate method with call scope"""
        payload = self.get_payload(prompt=prompt, stream=False, **kwargs)
        out = self.client.get_req_generation(self.model, payload=payload)
        return out

    def get_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator:
        """Call to client stream method with call scope"""
        payload = self.get_payload(prompt=prompt, stream=True, **kwargs)
        return self.client.get_req_stream(self.model, payload=payload)

    def get_astream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator:
        """Call to client astream methods with call scope"""
        payload = self.get_payload(prompt=prompt, stream=True, **kwargs)
        return self.client.get_req_astream(self.model, payload=payload)

    def get_payload(self, prompt: str, **kwargs: Any) -> dict:
        """Generates payload for the _NVIDIAClient API to send to service."""
        attr_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "bad": self.bad,
            "stop": self.stop,
            "labels": self.labels,
        }            
        default_kwargs = self.default_kwargs({"client": self.__class__.__name__})
        attr_kwargs = {k: v for k, v in attr_kwargs.items() if v is not None}
        new_kwargs = {**default_kwargs, **attr_kwargs, **kwargs}
        return self.prep_payload(prompt=prompt, **new_kwargs)

    def prep_payload(self, prompt: str, **kwargs: Any) -> dict:
        """Prepares a message or list of messages for the payload"""
        if kwargs.get("stop") is None:
            if self.stop:
                kwargs["stop"] = self.stop
            else:
                kwargs.pop("stop")
        return {"prompt": prompt, **kwargs}
