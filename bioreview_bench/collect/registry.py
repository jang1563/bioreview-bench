"""Source registry for multi-source data collection.

Centralizes configuration for each data source (eLife, PLOS, F1000, etc.)
so that the update pipeline can treat all sources uniformly.

Each source is defined by a SourceConfig that specifies:
- Which collector class to use
- Default parameters (start_date, subjects, output filename)
- How to generate article IDs from collector metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a single data source."""

    name: str                                   # "elife", "plos", "f1000"
    collector_module: str                       # "bioreview_bench.collect.elife"
    collector_class: str                        # "ELifeCollector"
    source_literal: str                         # matches OpenPeerReviewEntry.source
    id_prefix: str                              # "elife:", "plos:", "f1000:"
    default_start_date: str                     # "2018-01-01"
    default_subjects: list[str] = field(default_factory=list)
    output_filename: str = ""                   # "elife_v1.1.jsonl"
    collector_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_collector_class(self) -> type:
        """Dynamically import and return the collector class."""
        import importlib
        mod = importlib.import_module(self.collector_module)
        return getattr(mod, self.collector_class)

    def make_article_id(self, raw_id: str) -> str:
        """Create a full article ID like 'elife:87528' from a raw ID."""
        return f"{self.id_prefix}{raw_id}"


# ── Source Registry ────────────────────────────────────────────────

SOURCE_REGISTRY: dict[str, SourceConfig] = {
    "elife": SourceConfig(
        name="elife",
        collector_module="bioreview_bench.collect.elife",
        collector_class="ELifeCollector",
        source_literal="elife",
        id_prefix="elife:",
        default_start_date="2018-01-01",
        default_subjects=["genetics-genomics", "cell-biology", "neuroscience"],
        output_filename="elife_v1.1.jsonl",
    ),
    "plos": SourceConfig(
        name="plos",
        collector_module="bioreview_bench.collect.plos",
        collector_class="PLOSCollector",
        source_literal="plos",
        id_prefix="plos:",
        default_start_date="2018-01-01",
        default_subjects=[],
        output_filename="plos_v1.jsonl",
        collector_kwargs={
            "journals": ["PLoSBiology", "PLoSGenetics", "PLoSCompBiol"],
        },
    ),
    "f1000": SourceConfig(
        name="f1000",
        collector_module="bioreview_bench.collect.f1000",
        collector_class="F1000Collector",
        source_literal="f1000",
        id_prefix="f1000:",
        default_start_date="2013-01-01",
        default_subjects=[],
        output_filename="f1000_v1.jsonl",
        collector_kwargs={
            "journals": ["F1000Research"],
        },
    ),
    "nature": SourceConfig(
        name="nature",
        collector_module="bioreview_bench.collect.nature",
        collector_class="NatureCollector",
        source_literal="nature",
        id_prefix="nature:",
        default_start_date="2022-01-01",
        default_subjects=[],
        output_filename="nature_v1.jsonl",
    ),
    "peerj": SourceConfig(
        name="peerj",
        collector_module="bioreview_bench.collect.peerj",
        collector_class="PeerJCollector",
        source_literal="peerj",
        id_prefix="peerj:",
        default_start_date="2013-01-01",
        default_subjects=[],
        output_filename="peerj_v1.jsonl",
    ),
}


def get_source_config(name: str) -> SourceConfig:
    """Get a SourceConfig by name. Raises KeyError if not found."""
    if name not in SOURCE_REGISTRY:
        valid = ", ".join(sorted(SOURCE_REGISTRY))
        raise KeyError(f"Unknown source '{name}'. Valid sources: {valid}")
    return SOURCE_REGISTRY[name]


def list_sources() -> list[str]:
    """Return sorted list of registered source names."""
    return sorted(SOURCE_REGISTRY)
