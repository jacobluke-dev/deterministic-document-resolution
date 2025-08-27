from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
key_id_var: ContextVar[Optional[str]] = ContextVar("key_id", default=None)
path_var: ContextVar[Optional[str]] = ContextVar("path", default=None)
method_var: ContextVar[Optional[str]] = ContextVar("method", default=None)
client_ip_var: ContextVar[Optional[str]] = ContextVar("client_ip", default=None)

def set_request_context(*, request_id=None, key_id=None, path=None, method=None, client_ip=None) -> None:
    if request_id is not None: request_id_var.set(request_id)
    if key_id is not None:     key_id_var.set(key_id)
    if path is not None:       path_var.set(path)
    if method is not None:     method_var.set(method)
    if client_ip is not None:  client_ip_var.set(client_ip)
