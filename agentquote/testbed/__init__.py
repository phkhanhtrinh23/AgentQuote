from .bigquery import BigQueryTestbed
from .earthengine import EarthEngineTestbed


def get_testbed(platform: str, **kw):
    if platform in ("earthengine", "ee"):
        return EarthEngineTestbed(**kw)
    if platform in ("bigquery", "bq", "sql"):
        return BigQueryTestbed(**{k: v for k, v in kw.items() if k in ("seed", "scale", "n_max")})
    raise ValueError(platform)


__all__ = ["EarthEngineTestbed", "BigQueryTestbed", "get_testbed"]
