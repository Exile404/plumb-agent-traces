from __future__ import annotations

from pathlib import Path

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Tracer

from plumb.tracing.jsonl_exporter import JSONLSpanExporter

DEFAULT_TRACES_DIR = Path("data/raw_traces")


def init_tracing(
    run_id: str, traces_dir: str | Path = DEFAULT_TRACES_DIR
) -> tuple[Tracer, TracerProvider, Path]:
    """Isolated provider writing spans to data/raw_traces/<run_id>.jsonl.

    Uses a local provider, not the global one, to avoid clashing with any
    OTel state vLLM may set. Call provider.shutdown() when the run ends to flush.
    """
    out_path = Path(traces_dir) / f"{run_id}.jsonl"
    resource = Resource.create({"service.name": "plumb-agent", "plumb.run_id": run_id})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(JSONLSpanExporter(out_path)))
    return provider.get_tracer("plumb"), provider, out_path