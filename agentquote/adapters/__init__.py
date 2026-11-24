from .base import DomainAdapter
from .bigquery import BigQueryAdapter
from .earthengine import EarthEngineAdapter


def get_adapter(platform: str, **kw) -> DomainAdapter:
    if platform in ("earthengine", "ee"):
        return EarthEngineAdapter(**kw)
    if platform in ("bigquery", "bq", "sql"):
        return BigQueryAdapter(**kw)
    raise ValueError(f"unknown platform {platform}")


__all__ = ["DomainAdapter", "EarthEngineAdapter", "BigQueryAdapter", "get_adapter"]
