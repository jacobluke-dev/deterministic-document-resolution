from collections.abc import Callable
from typing import Any

MapperFn = Callable[[dict[str, Any]], dict[str, Any]]
