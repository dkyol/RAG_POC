from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
import time
from copy import deepcopy
from collections import defaultdict
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import aiohttp
import requests
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    PrivateAttr,
    SecretStr,
    root_validator,
)
from requests.models import Response

from langchain_nvidia_ai_endpoints._statics import Model
import langchain_nvidia_ai_endpoints._statics as specs

logger = logging.getLogger(__name__)

_MODE_TYPE = Literal["catalog", "nvidia", "nim", "open", "openai", "nvcf"]


def default_payload_fn(payload: dict) -> dict:
    return payload


def make_safe(d, sensitive_keys=["Authorization"]):
    safe = deepcopy(dict(d))
    stack = [safe]
    while stack:
        entry = stack.pop()
        if isinstance(entry, dict):
            for key in sensitive_keys:
                if key in entry: 
                    entry[key] = SecretStr(key)
            stack += [v for v in entry.values() if isinstance(v, (dict, list, tuple))]
        elif isinstance(entry, (list, tuple)):
            stack += [v for v in entry if isinstance(v, (dict, list, tuple))]
    return safe


class ClientTape():

    TapePool: Dict[int, Any] = {}

    def __init__(self):
        self.trace = defaultdict(lambda: [])
        self.idx = len(self.__class__.TapePool)

    def __enter__(self):
        self.__class__.TapePool[self.idx] = self
        return self

    def __exit__(self, *args: List, **kwargs: Any):
        del self.__class__.TapePool[self.idx]

    def clear(self):
        self.trace.clear()

    @classmethod
    def record(cls, key, value, idx=None, safe=False):
        output = value if not safe else make_safe(value)
        tapes = cls.TapePool.values() if idx is None else cls.TapePool.get(idx)
        for tape in tapes:
            tape.trace[key] += [output]
        return output


class BaseClient(BaseModel, ABC):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """

    # todo: add a validator for requests.Response (last_response attribute) and
    #       remove arbitrary_types_allowed=True
    class Config:
        arbitrary_types_allowed = True

    ## Core defaults. These probably should not be changed
    _api_key_var = "NVIDIA_API_KEY"
    _base_url_var = "NVIDIA_BASE_URL"
    base_url: str = Field(description="Base URL for standard inference")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)
    model_specs: Dict[str, Dict] = Field(specs.MODEL_SPECS)
    endpoints: dict

    api_key: SecretStr = Field(..., description="API Key for service of choice")

    ## Generation arguments
    timeout: float = Field(60, ge=0, description="Timeout for waiting on response (s)")
    interval: float = Field(0.02, ge=0, description="Interval for pulling response")
    last_inputs: dict = Field({}, description="Last inputs sent over to the server")
    last_response: Response = Field(
        None, description="Last response sent from the server"
    )
    payload_fn: Callable = Field(
        default_payload_fn, description="Function to process payload"
    )
    headers_tmpl: dict = Field(
        ...,
        description="Headers template for API calls."
        " Should contain `call` and `stream` keys.",
    )
    _available_functions: Optional[List[dict]] = PrivateAttr(default=None)
    _available_models: Optional[dict] = PrivateAttr(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": self._api_key_var}

    @property
    def headers(self) -> dict:
        """Return headers with API key injected"""
        headers_ = deepcopy(self.headers_tmpl)
        for header in headers_.values():
            if "{api_key}" in header["Authorization"]:
                header["Authorization"] = header["Authorization"].format(
                    api_key=self.api_key.get_secret_value(),
                )
        return headers_

    @root_validator(pre=True)
    def validate_model(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update model arguments, including API key and formatting"""
        api_key_var = values.pop("api_key_var", cls._api_key_var)
        base_url_var = values.pop("base_url_var", cls._base_url_var)
        values["api_key"] = (
            values.get(api_key_var.lower())
            or values.get("api_key")
            or os.getenv(api_key_var)
            or ""
        )
        values["base_url"] = (
            values.get(base_url_var.lower())
            or values.get("base_url")
            or os.getenv(base_url_var)
            or ""
        )
        if not values.get("base_url"):
            values.pop("base_url")
        values["is_staging"] = "nvapi-stg-" in values["api_key"]
        if "headers_tmpl" not in values:
            call_kvs = {
                "Accept": "application/json",
            }
            stream_kvs = {
                "Accept": "text/event-stream",
                "content-type": "application/json",
            }
            shared_kvs = {
                "Authorization": "Bearer {api_key}",
                "User-Agent": "langchain-nvidia-ai-endpoints",
            }
            values["headers_tmpl"] = {
                "call": {**call_kvs, **shared_kvs},
                "stream": {**stream_kvs, **shared_kvs},
            }
        return values

    @abstractmethod
    def parse_model_response(self):
        pass
    
    @property
    def available_models(self) -> dict:
        """List the available models that can be invoked."""
        # live_fns = self.available_functions
        # if "status" in live_fns[0]:
        #     live_fns = [v for v in live_fns if v.get("status") == "ACTIVE"]
        #     self._available_models = {v["name"]: v["id"] for v in live_fns}
        # else:
        #     self._available_models = {v.get("id"): v.get("owned_by") for v in live_fns}
        if not self.endpoints.get("models"):
            raise ValueError("No models endpoint found, so cannot retrieve model list.")
        self._available_models = self.parse_model_response()
        return self._available_models

    def reset_method_cache(self) -> None:
        """Reset method cache to force re-fetch of available functions"""
        self._available_functions = None
        self._available_models = None

    ####################################################################################
    ## Core utilities for posting and getting from NV Endpoints

    def _post(
        self,
        invoke_url: str,
        payload: Optional[dict] = {},
    ) -> Tuple[Response, Any]:
        """Method for posting to the AI Foundation Model Function API."""
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "json": self.payload_fn(payload),
            "stream": False,
        }
        self.last_inputs = ClientTape.record("request", last_inputs, safe=True)
        session = self.get_session_fn()
        response = session.post(**last_inputs)
        self.last_response = ClientTape.record("response", response)
        self._try_raise(response)
        return response, session

    def _get(
        self,
        invoke_url: str,
        payload: Optional[dict] = {},
    ) -> Tuple[Response, Any]:
        """Method for getting from the AI Foundation Model Function API."""
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["call"],
            "stream": False,
        }
        if payload:
            last_inputs["json"] = self.payload_fn(payload)
        self.last_inputs = ClientTape.record("request", last_inputs, safe=True)
        session = self.get_session_fn()
        response = session.get(**last_inputs)
        self.last_response = ClientTape.record("response", response)
        self._try_raise(response)
        return response, session

    def _wait(self, response: Response, session: Any) -> Response:
        """Wait for a response from API after an initial response is made"""
        start_time = time.time()
        while response.status_code == 202:
            time.sleep(self.interval)
            if (time.time() - start_time) > self.timeout:
                raise TimeoutError(
                    f"Timeout reached without a successful response."
                    f"\nLast response: {str(response)}"
                )
            request_id = response.headers.get("NVCF-REQID", "")
            endpoint_args = {"base_url": self.base_url, "request_id": request_id}
            self.last_response = response = session.get(
                self.endpoints["status"].format(**endpoint_args),
                headers=self.headers["call"],
            )
        self._try_raise(response)
        return response

    def _try_raise(self, response: Response) -> None:
        """Try to raise an error from a response"""
        try:
            response.raise_for_status()
        except requests.HTTPError:
            rd = response.__dict__
            try:
                rd_temp = response.json()
                if "detail" in rd and "reqId" in rd.get("detail", ""):
                    rd_buf = "- " + str(rd["detail"])
                    rd_buf = rd_buf.replace(": ", ", Error: ").replace(", ", "\n- ")
                    rd_temp["detail"] = rd_buf
                rd = {**rd_temp, **rd}
            except json.JSONDecodeError:
                if "status_code" in rd:
                    if "headers" in rd and "WWW-Authenticate" in rd["headers"]:
                        rd["detail"] = rd.get("headers").get("WWW-Authenticate")
                        rd["detail"] = rd["detail"].replace(", ", "\n")
                else:
                    rd_temp = rd.pop("_content")
                    if isinstance(rd_temp, bytes):
                        rd_temp = rd_temp.decode("utf-8")[5:]  ## remove "data:" prefix
                    try:
                        rd_temp = json.loads(rd_temp)
                    except Exception:
                        rd_temp = {"detail": rd_temp}
                    rd = {**rd_temp, **rd}
            status = rd.get("status") or rd.get("status_code") or "###"
            title = (
                rd.get("title")
                or rd.get("error", {}).get("code")
                or rd.get("reason")
                or "Unknown Error"
            )
            header = f"[{status}] {title}"
            body = ""
            if "requestId" in rd:
                if "detail" in rd:
                    body += f"{rd['detail']}\n"
                body += "RequestID: " + rd["requestId"]
            else:
                body = str(
                    rd.get("error", {}).get("message")
                    or rd.get("reason")
                    or rd.get("detail") 
                    or rd
                )
            if str(status) == "401":
                body += "\nPlease check or regenerate your API key."
            # todo: raise as an HTTPError
            raise Exception(f"{header}\n{body}") from None

    ####################################################################################
    ## Simple query interface to show the set of model options

    def query(
        self,
        invoke_url: str,
        payload: Optional[dict] = None,
        request: str = "get",
    ) -> dict:
        """Simple method for an end-to-end get query. Returns result dictionary"""
        if request == "get":
            response, session = self._get(invoke_url, payload)
        else:
            response, session = self._post(invoke_url, payload)
        response = self._wait(response, session)
        output = self._process_response(response)[0]
        return output

    def _process_response(self, response: Union[str, Response]) -> List[dict]:
        """General-purpose response processing for single responses and streams"""
        if hasattr(response, "json"):  ## For single response (i.e. non-streaming)
            try:
                return [response.json()]
            except json.JSONDecodeError:
                response = str(response.__dict__)
        if isinstance(response, str):  ## For set of responses (i.e. streaming)
            msg_list = []
            for msg in response.split("\n\n"):
                if "{" not in msg:
                    continue
                msg_list += [json.loads(msg[msg.find("{") :])]
            return msg_list
        raise ValueError(f"Received ill-formed response: {response}")
    
    def _get_invoke_url(
        self,
        model_name: Optional[str] = None,
        invoke_url: Optional[str] = None,
        endpoint: str = "",
    ) -> str:
        """Helper method to get invoke URL from a model name, URL, or endpoint stub"""
        if not invoke_url:            
            ## Try to override call args by statics/model discovery
            available_models = self.available_models
            mspec = available_models.get(model_name, {})
            assert isinstance(mspec, dict), f"Expected dict, received {mspec}"
            base_url = mspec.get("base_url", self.base_url)
            path_str = mspec.get("infer_path", self.endpoints.get(endpoint, ""))
            model_id = ""
            
            if not path_str:
                raise ValueError(f"Unknown endpoint referenced {endpoint} provided")
            
            if "{model_id}" in path_str:
                if not model_name:
                    raise ValueError("URL or model name must be specified to invoke")
                if model_name not in available_models:
                    valid_models = [f"{k} - {v}" for k, v in available_models.items()]
                    valid_models = "\n".join(valid_models)
                    raise ValueError(
                        f"Unknown model name {model_name} specified."
                        f"\nAvailable models are:\n{valid_models}"
                    )
                model_id = available_models.get(model_name).get("model_id")
                if not model_id:
                    raise ValueError(f"No model_id discovered for model '{model_name}'")            
            path_kws = {"base_url": base_url, "model_id": model_id, "model_name": model_name}
            invoke_url = path_str.format(**path_kws)

        if not invoke_url:
            raise ValueError("URL or model name must be specified to invoke")

        return invoke_url

    ####################################################################################
    ## Generation interface to allow users to generate new values from endpoints

    def get_req(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        endpoint: str = "",
    ) -> Response:
        """Post to the API."""
        invoke_url = self._get_invoke_url(model_name, invoke_url, endpoint=endpoint)
        if payload.get("stream", False) is True:
            payload = {**payload, "stream": False}
        response, session = self._post(invoke_url, payload)
        return self._wait(response, session)

    def get_req_generation(
        self,
        model_name: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        endpoint: str = "infer",
    ) -> dict:
        """Method for an end-to-end post query with NVE post-processing."""
        invoke_url = self._get_invoke_url(model_name, invoke_url, endpoint=endpoint)
        response = self.get_req(model_name, payload, invoke_url)
        output = self.postprocess(response)
        return output

    def postprocess(
        self, response: Union[str, Response]
    ) -> Tuple[dict, bool]:
        """Parses a response from the AI Foundation Model Function API.
        """
        msg_list = self._process_response(response)
        responses = self._aggregate_msgs(msg_list)
        return responses

    def _aggregate_msgs(self, msg_list: Sequence[dict]) -> Tuple[dict, bool]:
        """Dig out relevant details of aggregated message"""
        content_buffer: List[Dict[str, Any]] = []
        content_holder: List[Dict[Any, Any]] = []
        stopped_holder: List[bool] = []
        usage_holder: Dict[Any, Any] = dict()  ####
        for response in msg_list:
            # usage_holder = response.get("usage", {})  ####
            if "choices" in response:
                ## Tease out ['choices'][0]...['delta'/'message']
                choices = response.get("choices", [{}])
                for i, msg in enumerate(choices):
                    stopped_holder += [msg.get("finish_reason", "") == "stop"]
                    msg = msg.get("delta", msg.get("message", msg.get("text", "")))
                    if not isinstance(msg, dict):
                        msg = {"content": msg}
                    content_holder += [msg]
                for i, msg in enumerate(content_holder):
                    content_buffer += [{}]
                    for k, v in msg.items():
                        if k in ("content",) and k in content_buffer:
                            content_buffer[-1][k] += v
                        else:
                            content_buffer[-1][k] = v
                        if stopped_holder[i]:
                            break
        content_holder = [{**h, **b} for h, b in zip(content_holder, content_buffer)]
        if content_holder:
            response['cumulative'] = content_holder
        return response

    ####################################################################################
    ## Streaming interface to allow you to iterate through progressive generations

    def get_req_stream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        endpoint: str = "infer",
    ) -> Iterator:
        invoke_url = self._get_invoke_url(model, invoke_url, endpoint=endpoint)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": self.payload_fn(payload),
            "stream": True,
        }
        self.last_inputs = ClientTape.record("request", last_inputs, safe=True)
        session = self.get_session_fn()
        response = session.post(**last_inputs)
        self.last_response = ClientTape.record("response", response)
        self._try_raise(response)
        call = self.copy()

        def out_gen() -> Generator[dict, Any, Any]:
            ## Good for client, since it allows self.last_inputs
            for line in response.iter_lines():
                if line and line.strip() != b"data: [DONE]":
                    line = line.decode("utf-8")
                    msg = call.postprocess(line)
                    yield msg
                self._try_raise(response)

        return (r for r in out_gen())

    ####################################################################################
    ## Asynchronous streaming interface to allow multiple generations to happen at once.

    async def get_req_astream(
        self,
        model: Optional[str] = None,
        payload: dict = {},
        invoke_url: Optional[str] = None,
        endpoint: str = "infer",
    ) -> AsyncIterator:
        invoke_url = self._get_invoke_url(model, invoke_url, endpoint=endpoint)
        if payload.get("stream", True) is False:
            payload = {**payload, "stream": True}
        last_inputs = {
            "url": invoke_url,
            "headers": self.headers["stream"],
            "json": self.payload_fn(payload),
        }
        self.last_inputs = ClientTape.record("request", last_inputs, safe=True)
        async with self.get_asession_fn() as session:
            async with session.post(**last_inputs) as response:
                self._try_raise(response)
                async for line in response.content.iter_any():
                    if line and line.strip() != b"data: [DONE]":
                        line = line.decode("utf-8")
                        msg = self.postprocess(line)
                        yield msg



class NVCFClient(BaseClient):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """

    ## Core defaults. These probably should not be changed
    base_url: str = Field("https://api.nvcf.nvidia.com/v2/nvcf")
    endpoints: dict = Field(
        {
            "infer": "{base_url}/pexec/functions/{model_id}",
            "status": "{base_url}/pexec/status/{request_id}",
            "models": "{base_url}/functions",
        }
    )

    is_staging: bool = Field(False, description="Whether to use staging API")

    def parse_model_response(self):
        """List the available functions that can be invoked."""
        try:
            invoke_url = self.endpoints.get("models", "").format(base_url=self.base_url)
            response = self.query(invoke_url)
        except Exception as e:
            raise ValueError(f"Failed to query model endpoint {invoke_url}.\n{e}")
        model_list = response.get("functions")
        if not isinstance(model_list, list):
            raise ValueError(
                f"Unexpected response when querying {invoke_url}\n{query_res}"
            )
        model_list = [v for v in model_list if v.get("status") == "ACTIVE"]
        return {v["name"]: {"model_id": v["id"]} for v in model_list}


class OpenClient(BaseClient):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """
    ## Core defaults. These probably should not be changed
    base_url: str = Field("https://integrate.api.nvidia.com/v1")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)
    endpoints: dict = Field(
        {
            "infer": "{base_url}/chat/completions",
            "models": "{base_url}/models",
        }
    )

    def parse_model_response(self):
        """List the available functions that can be invoked."""
        try:
            invoke_url = self.endpoints.get("models", "").format(base_url=self.base_url)
            response = self.query(invoke_url)
        except Exception as e:
            raise ValueError(f"Failed to query model endpoint {invoke_url}.\n{e}")
        model_list = response.get("data")
        if not isinstance(model_list, list):
            raise ValueError(
                f"Unexpected response when querying {invoke_url}\n{query_res}"
            )
        out = {v.get("id"): {**v, **self.model_specs.get(v.get("id"), {})} for v in model_list}
        ## TODO: Adjust scope/power of NVIDIA_BASE_URL
        if os.environ.get("NVIDIA_BASE_URL", "") == self.base_url:
            for v in out.values():
                if v.get("base_url"):
                    v["base_url"] = self.base_url
        return out


class StaticClient(BaseClient):

    """
    Underlying Client for interacting with the AI Foundation Model Function API.
    Leveraged by the NVIDIABaseModel to provide a simple requests-oriented interface.
    Direct abstraction over NGC-recommended streaming/non-streaming Python solutions.

    NOTE: Models in the playground does not currently support raw text continuation.
    """
    ## Core defaults. These probably should not be changed
    base_url: str = Field("https://integrate.api.nvidia.com/v1")
    get_session_fn: Callable = Field(requests.Session)
    get_asession_fn: Callable = Field(aiohttp.ClientSession)
    endpoints: dict = Field(
        {
            "infer": "{base_url}/chat/completions",
            "models": "{base_url}/models",
        }
    )

    def parse_model_response(self):
        """List the available functions that can be invoked."""
        return {k: v for k, v in self.model_specs.items()}


class BaseNVIDIA(BaseModel):
    """
    Higher-Level AI Foundation Model Function API Client with argument defaults.
    Is subclassed by ChatNVIDIA to provide a simple LangChain interface.
    """

    client: BaseClient = Field()

    _default_model: str = ""
    _default_mode: str = "nvidia"
    model: str = Field(description="Name of the model to invoke")
    curr_mode: _MODE_TYPE = Field(_default_mode)

    ####################################################################################

    @root_validator(pre=True)
    def validate_client(cls, values: Any) -> Any:
        """Validate and update client arguments, including API key and formatting"""
        if not values.get("model"):
            values["model"] = cls._default_model
            assert values["model"], "No model given, with no default to fall back on."
        
        model_specs = {**values.get("model_specs", {}), **specs.MODEL_SPECS}
        default_mode = model_specs.get(values['model'], {}).get("mode")
        values["curr_mode"] = (
            values.pop("mode", None) 
            or values.get("curr_mode")
            or os.environ.get("NVIDIA_DEFAULT_MODE")
            or default_mode
        )
        ## TODO: This try-override fails when NVIDIA_DEFAULT_MODE is defined. 
        ## (since environment assumes advanced use cases) and isn't too robust. 
        ## Good to corroborate with model list first?
        if not values["curr_mode"]:
            old_name = values['model']
            for name in (f"playground_{old_name}", f"ai-{old_name}"):
                if name in model_specs:
                    values['model'] = name
                    values['curr_mode'] = model_specs.get(name).get("mode")
        if not values["curr_mode"]:
            values["curr_mode"] = cls._default_mode
        if not values.get("client"):
            dummy_self = cls(client=StaticClient())
            mode_kw = {
                "mode": values["curr_mode"],
                "base_url": values.pop("base_url", os.environ.get("NVIDIA_BASE_URL")),
            }
            values["client"] = dummy_self.mode(**values, **mode_kw).client
            
        # the only model that doesn't support a stream parameter is kosmos_2.
        # to address this, we'll use the payload_fn to remove the stream parameter for
        # kosmos_2. if a user tries to set their own payload_fn, this patch will be
        # overwritten.
        # todo: get kosmos_2 api updated to support stream parameter
        if values["model"] == "kosmos_2":

            def kosmos_patch(payload: dict) -> dict:
                payload.pop("stream", None)
                return payload

            values["client"].payload_fn = kosmos_patch
        
        return values
    
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": self.client._api_key_var}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if getattr(self.client, "base_url"):
            attributes["base_url"] = self.client.base_url

        if self.model:
            attributes["model"] = self.model

        if getattr(self.client, "endpoints"):
            attributes["endpoints"] = self.client.endpoints

        return attributes

    @property
    def available_functions(self) -> List[Model]:
        """Map the available functions that can be invoked."""
        return self.available_models

    @property
    def available_models(self) -> List[Model]:
        """Map the available models that can be invoked."""
        return self.get_available_models(client=self)

    @classmethod
    def get_available_functions(cls, *args: List, **kwargs: Any) -> List[Model]:
        """Map the available functions that can be invoked. Callable from class"""
        return cls.get_available_models(*args, **kwargs)
        

    @classmethod
    def get_available_models(
        cls,
        mode: Optional[_MODE_TYPE] = None,
        client: Any = None,
        list_all: bool = False,
        list_none: bool = None,
        list_meta: bool = False,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Model]:
        """Map the available models that can be invoked. Callable from class"""
        self = client if isinstance(client, BaseModel) else cls(**kwargs).mode(mode, **kwargs)
        needs_mode = (mode is not None and mode != self.curr_mode)
        self = self if not needs_mode else self.mode(mode, **kwargs)
        client = self.client
        mode = mode or self.curr_mode
        list_none = list_none if list_none is not None else (mode in ("open", "nim")) 
        
        client.reset_method_cache()
        out = sorted(
            [
                Model(id=k, **client.model_specs.get(k, {}))
                for k, v in client.available_models.items()
            ],
            key=lambda x: f"{x.metadata.client_args.get('client') or 'Z'}{x.id}{cls}",
        )
        if not filter:
            filter = [{"client": cls.__name__}]
        elif isinstance(filter, str):
            filter = [filter]
        elif isinstance(filter, dict):
            filter = list(filter.items())
        if not list_all:
            out = [
                m for m in out 
                if (list_none is True and m.model_type is None)
                or all(
                    (isinstance(f, str) and str(f) in str(m)) or
                    (isinstance(f, dict) and any(
                        (key in f and f.get(key) == value) or
                        (isinstance(value, list) and f.get(key) in value)
                        for key, value in [
                            *m.metadata.client_args.items(),
                            *m.metadata.infer_args.items(),
                            *m.dict().items(),
                        ]
                    ))
                    for f in filter
                )
            ]
        if not list_meta:
            for model in out:
                del model.metadata
        return out

    def default_kwargs(self, type: str) -> dict:
        ## TODO: Type filtering sliminates some true positives. Bypassing with list_all
        models = self.get_available_models(client=self, filter=type, list_all=True, show_clientless=True, list_meta=True)
        matches = [model for model in models if model.id == self.model]
        if self.curr_mode in ("openai",):
            return {'model': self.model}
        if not matches:
            raise ValueError(f"Model '{self.model}' not found")
        out = matches[0].metadata.infer_args
        out["model"] = out.pop("model_name", self.model)
        return out

    def get_binding_model(self) -> Optional[str]:
        """Get the model to bind to the client as default payload argument"""
        # if a model is configured with a model_name, always use that
        # todo: move from search of available_models to a Model property
        matches = [model for model in self.available_models if model.id == self.model]
        if matches:
            if matches[0].id:
                return matches[0].id
        if self.curr_mode == "nvcf":
            return ""
        return self.model

    def mode(
        self,
        mode: Optional[_MODE_TYPE] = "nvidia",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        model_specs: dict = {},
        # infer_path: Optional[str] = None,
        # models_path: Optional[str] = "{base_url}/models",
        force_mode: bool = False,
        force_clone: bool = True,
        **kwargs: Any,
    ) -> Any:  # todo: in python 3.11+ this should be typing.Self
        """Return a client swapped to a different mode"""
        if isinstance(self, str):
            raise ValueError("Please construct the model before calling mode()")
        out = self if not force_clone else deepcopy(self)

        if mode is None:
            return out

        catalog_base = "https://integrate.api.nvidia.com/v1"
        openai_base = "https://api.openai.com/v1"  ## OpenAI Main URL
        nvcf_base = "https://api.nvcf.nvidia.com/v2/nvcf"  ## NVCF Main URL

        if mode == "nvcf":
            ## Classic support for nvcf-backed foundation+ model endpoints.
            mspecs = model_specs or specs.NVCF_SPECS
            out.client = NVCFClient(base_url=nvcf_base, model_specs=mspecs)

        elif mode == "nvidia" or mode == "catalog":
            ## NVIDIA API Catalog Integration: OpenAPI-spec gateway
            mspecs = model_specs or specs.CATALOG_SPECS
            out.client = StaticClient(base_url=catalog_base, api_key=api_key, model_specs=mspecs)

        elif mode == "open" or mode == "nim":
            ## OpenAPI-style specs to connect to arbitrary running NIM-LLM instance
            assert base_url, "Base URL must be specified for open/nim mode"
            mspecs = model_specs or specs.OPEN_SPECS
            out.client = OpenClient(base_url=base_url, api_key=api_key, model_specs=mspecs)

        elif mode == "openai":
            ## OpenAI-style specification to connect to OpenAI endpoints
            mspecs = model_specs or specs.OPEN_SPECS
            out.client = OpenClient(base_url=openai_base, api_key=api_key, model_specs=mspecs, api_key_var="OPENAI_API_KEY")

        else:
            raise ValueError(f"Unknown mode: `{mode}`. Expected one of {_MODE_TYPE}.")

        out.client.reset_method_cache()

        return out
