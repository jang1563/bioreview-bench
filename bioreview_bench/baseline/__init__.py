from typing import TYPE_CHECKING, Any

from .lexical import BM25ConcernRetriever

if TYPE_CHECKING:
    from .reviewer import BaselineReviewer

__all__ = ["BaselineReviewer", "BM25ConcernRetriever"]


def __getattr__(name: str) -> Any:
    """Load provider-dependent baseline code only when it is requested."""
    if name == "BaselineReviewer":
        from .reviewer import BaselineReviewer

        return BaselineReviewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
