from __future__ import annotations

from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from plumb.schemas import RawSpan


def _span_to_raw(span: ReadableSpan) -> RawSpan:
    ctx = span.get_span_context()
    parent = span.parent
    return RawSpan(
        trace_id=f"{ctx.trace_id:032x}",
        span_id=f"{ctx.span_id:016x}",
        parent_span_id=f"{parent.span_id:016x}" if parent else None,
        name=span.name,
        kind=span.kind.name,
        start_time_ns=span.start_time,
        end_time_ns=span.end_time,
        attributes=dict(span.attributes or {}),
    )


class JSONLSpanExporter(SpanExporter):
    """Append each span to a JSON lines file as a RawSpan record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # start fresh per run; re-runs overwrite, never append

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a", encoding="utf-8") as handle:
            for span in spans:
                handle.write(_span_to_raw(span).model_dump_json() + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True