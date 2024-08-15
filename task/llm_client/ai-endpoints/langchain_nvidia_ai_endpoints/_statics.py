import os
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator, Field


class Metadata(BaseModel):
    infer_args: dict = Field({})
    client_args: dict = Field({})
        

class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    metadata: Optional[Metadata] = None
    # path: str

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_required_field_names = {field.alias for field in cls.__fields__.values() if field.alias != 'metadata'}
        client_args = ["client", "base_url", "infer_path", "api_key_var", "api_type", "mode", "alternative"]
        client_kw: Dict[str, Any] = {}
        infer_kw: Dict[str, Any] = {}
        for field_name in list(values):
            if field_name in client_args:
                client_kw[field_name] = values.pop(field_name)
            elif field_name not in all_required_field_names:
                infer_kw[field_name] = values.pop(field_name)
        values['metadata'] = Metadata(client_args=client_kw, infer_args=infer_kw)
        return values


NVCF_PG_SPECS = {
}

NVCF_AI_SPECS = {
}

CATALOG_SPECS = {
    'databricks/dbrx-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'google/codegemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/recurrentgemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/codellama-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama2-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama-3.1-70b-instruct': {'model_type': 'chat', 'max_tokens': 8096},
    'meta/llama-3.1-8b-instruct': {'model_type': 'chat', 'max_tokens': 8096},
    'meta/llama3-70b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama3-8b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-128k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-4k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-7b-instruct-v0.2': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-large': {'model_type': 'chat', 'max_tokens': 2048}, 
    'mistralai/mixtral-8x22b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    ## TODO: Very temporary behavior. Expect it to change
    'mistralai/mixtral-8x22b-v0.1': {'model_type': 'completion', 'max_tokens': 2048, 'infer_path': '{base_url}/chat/completions'},
    'mistralai/mixtral-8x7b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    'snowflake/arctic': {'model_type': 'chat', 'max_tokens': 2048},
    'nvidia/nv-embedqa-mistral-7b-v2': {'model_type': 'embedding'},
    'nvidia/nv-embed-v1': {'model_type': 'embedding'},
    'nvidia/nv-embedqa-e5-v5': {'model_type': 'embedding'},
    'nvidia/embed-qa-4': {'model_type': 'embedding'},
    "nvidia/rerank-qa-mistral-4b": {
        "model_type": "ranking",
        "model_name": "nv-rerank-qa-mistral-4b:1",
        'base_url': 'https://ai.api.nvidia.com/v1', 
        'infer_path': '{base_url}/retrieval/nvidia/reranking', 
    },
    'snowflake/arctic-embed-l': {'model_type': 'embedding'},
    "stabilityai/stable-diffusion-xl": {'model_type': 'genai'},
    "stabilityai/sdxl-turbo": {'model_type': 'genai'},
    "stabilityai/stable-video-diffusion": {'model_type': 'genai'},
}

OPENAI_SPECS = {
    "babbage-002": {"model_type": "completion"},
    "dall-e-2": {"model_type": "genai"},
    "dall-e-3": {"model_type": "genai"},
    "davinci-002": {"model_type": "completion"},
    "gpt-3.5-turbo-0125": {"model_type": "chat"},
    "gpt-3.5-turbo-0301": {"model_type": "chat"},
    "gpt-3.5-turbo-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-1106": {"model_type": "chat"},
    "gpt-3.5-turbo-16k-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-16k": {"model_type": "chat"},
    "gpt-3.5-turbo-instruct-0914": {"model_type": "completion"},
    "gpt-3.5-turbo-instruct": {"model_type": "completion"},
    "gpt-3.5-turbo": {"model_type": "chat"},
    "gpt-4": {"model_type": "chat"},
    "gpt-4-0125-preview": {"model_type": "chat"},
    "gpt-4-0613": {"model_type": "chat"},
    "gpt-4-1106-preview": {"model_type": "chat"},
    "gpt-4-1106-vision-preview": {"model_type": "chat"},
    "gpt-4-turbo": {"model_type": "chat"},
    "gpt-4-turbo-2024-04-09": {"model_type": "chat"},
    "gpt-4-turbo-preview": {"model_type": "chat"},
    "gpt-4-vision-preview": {"model_type": "chat"},
    "text-embedding-3-large": {"model_type": "embedding"},
    "text-embedding-3-small": {"model_type": "embedding"},
    "text-embedding-ada-002": {"model_type": "embedding"},
    "tts-1-1106": {"model_type": "tts"},
    "tts-1-hd-1106": {"model_type": "tts"},
    "tts-1-hd": {"model_type": "tts"},
    "tts-1": {"model_type": "tts"},
    "whisper-1": {"model_type": "asr"},
}

CLIENT_MAP = {
    "asr": "RunnableNVIDIA",
    "chat": "ChatNVIDIA",
    "classifier": "RunnableNVIDIA",
    "completion": "NVIDIA",
    "cuopt": "RunnableNVIDIA",
    "embedding": "NVIDIAEmbeddings",
    "vlm": "ChatNVIDIA",
    "genai": "RunnableNVIDIA",
    "qa": "ChatNVIDIA",
    "similarity": "RunnableNVIDIA",
    "translation": "RunnableNVIDIA",
    "tts": "RunnableNVIDIA",
    "ranking": "NVIDIARerank",
}

for mname, mspec in CATALOG_SPECS.items():
    base_url = ""
    infer_path = ""
    if mspec.get('model_type') == 'vlm':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/vlm/{model_name}'
    elif mspec.get('model_type') == 'genai':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/genai/{model_name}'
    elif mspec.get('model_type') == 'embeddings':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/retrieval/{model_name}'
    if base_url and 'base_url' not in mspec:
        mspec['base_url'] = base_url
    if infer_path and 'infer_path' not in mspec:
        mspec['infer_path'] = infer_path

tooled_models = ["gpt-4"]

SPEC_LIST = [CATALOG_SPECS, OPENAI_SPECS, NVCF_PG_SPECS, NVCF_AI_SPECS]
MODE_LIST = ["nvidia", "openai", "nvcf", "nvcf"]

for spec, mode in zip(SPEC_LIST, MODE_LIST):
    for mname, mspec in spec.items():
        mspec["mode"] = mspec.get("mode") or mode
        ## Default max_tokens for models
        if mspec.get('model_type') in ('chat', 'vlm'):
            if 'max_tokens' not in mspec:
                mspec['max_tokens'] = 1024
        ## Default Client enforcement
        mspec['client'] = mspec.get("client") or [CLIENT_MAP.get(mspec.get("model_type"))]
        if not isinstance(mspec['client'], list):
            mspec['client'] = [mspec['client']]
        if mname in tooled_models:
            mspec['client'] += [f"Tooled{client}" for client in mspec['client']]

OPEN_SPECS = {**CATALOG_SPECS, **OPENAI_SPECS}

for mname, mspec in OPEN_SPECS.items():
    if not mspec.get('infer_path'):
        model_type = mspec.get('model_type')
        if model_type == 'chat':
            mspec['infer_path'] = '{base_url}/chat/completions'
        if model_type == 'completion':
            mspec['infer_path'] = '{base_url}/completions'
        if model_type == 'embedding':
            mspec['infer_path'] = '{base_url}/embeddings'
        if model_type == 'genai':
            mspec['infer_path'] = '{base_url}/images/generations'

NVCF_SPECS = {**NVCF_PG_SPECS, **NVCF_AI_SPECS}

MODEL_SPECS = {**OPEN_SPECS, **NVCF_SPECS}