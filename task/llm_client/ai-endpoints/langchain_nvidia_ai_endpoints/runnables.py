"""Embeddings Components Derived from NVEModel/Embeddings"""
import base64
from functools import partial
from operator import itemgetter
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.runnables import Runnable, RunnableLambda
from PIL import Image

from langchain_nvidia_ai_endpoints._common import BaseNVIDIA


class MediaParser:
    """
    Handles conversion between base64 strings, bytes, and PIL Images, 
    and facilitates file operations for media content.
    """

    #### Primitive Methods
    
    @staticmethod
    def base64_to_bytes(base64_str: str) -> bytes:
        """Converts base64 string to bytes."""
        return base64.b64decode(base64_str)

    @staticmethod
    def bytes_to_base64(byte_data: bytes) -> str:
        """Converts bytes to base64 string."""
        return base64.b64encode(byte_data).decode('utf-8')

    @staticmethod
    def bytes_to_image(byte_data: bytes) -> Image.Image:
        """Converts byte data to a PIL Image object."""
        return Image.open(BytesIO(byte_data))

    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
        """Converts a PIL Image object to bytes in the specified format."""
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()

    @staticmethod
    def bytes_to_file(b8: bytes, file_path: str = "temp.mp4") -> None:
        """Writes byte data to a file specified by file_path."""
        with open(file_path, 'wb') as file:
            file.write(b8)

    @staticmethod
    def image_to_file(
        image: Image.Image, file_path: str = "temp.jpg", format: str = None,
    ) -> Image.Image:
        """Saves a PIL Image to a file, inferring format from the file extension if not specified."""
        if format is None:
            format = 'PNG' if file_path.lower().endswith('.png') else 'JPEG'
        image.save(file_path, format=format)
        return image

    def ForEach(fn) -> Runnable:
        return RunnableLambda(lambda d: [RunnableLambda(fn).invoke(x) for x in d])
        
    #### Composite Methods
    
    @staticmethod
    def b64_to_image(base64_str: str) -> Image.Image:
        out = MediaParser.base64_to_bytes(base64_str)
        out = MediaParser.bytes_to_image(out)
        return out

    @staticmethod
    def b64_to_image_file(
        base64_str: str, file_path: str = "temp.jpg", format: str = None,
    ) -> Image.Image:
        out = MediaParser.b64_to_image(base64_str)
        if file_path:
            out = MediaParser.image_to_file(out, file_path=file_path, format=format)
        return out
    
    @staticmethod
    def b64_to_video_file(
        base64_str: str, file_path: str = "temp.mp4",
    ) -> bytes:
        out = MediaParser.base64_to_bytes(base64_str)
        if file_path:
            out = MediaParser.bytes_to_file(out, file_path=file_path)
        return out


def ParseImage():
    def _internal(data):
        b64_strs = [a.get("base64") for a in data.get("artifacts")]
        return [Image.open(BytesIO(base64.decodebytes(bytes(a, "utf-8")))) for a in b64_strs]
    return RunnableLambda(_internal)


class RunnableNVIDIA(BaseNVIDIA, RunnableLambda):
    """NVIDIA's AI Foundation Retriever Question-Answering Asymmetric Model."""

    _default_model: str = "stabilityai/stable-diffusion-xl"
    model: str = Field(description="Name of the model to invoke")
    attr_kwargs: dict = Field({}, description="Payload arguments for request")

    @root_validator(pre=True)
    def get_extras(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_field_names = {field.name for field in cls.__fields__.values()}
        all_field_alias = {field.alias for field in cls.__fields__.values()}
        all_fields = all_field_names | all_field_alias
        infer_kw: Dict[str, Any] = {}
        for field_name in list(values):
            if field_name not in all_fields:
                infer_kw[field_name] = values.pop(field_name)
        values['attr_kwargs'] = infer_kw
        return values
    
    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "nvidia-image-gen-model"

    def func(self, kwargs: Any) -> str:
        """Run the Image Gen Model on the given prompt and input."""
        default_kwargs = self.default_kwargs({"client": self.__class__.__name__})
        default_kwargs.pop("model")
        payload = {**default_kwargs, **self.attr_kwargs, **kwargs}
        response = self.client.get_req_generation(self.model, payload=payload)
        return response

    def as_pil(self) -> Runnable:
        """Returns a model that outputs a PIL image by default, decoding from b64."""
        return self | ParseImage()
