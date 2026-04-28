"""Logging formatters used across all projects."""

import json
import logging

import numpy as np

from ssl_simulator.config import CONFIG


# ---------- helpers ----------
def _is_simple(v) -> bool:
    """A value is 'simple' if it can render on a single line."""
    if isinstance(v, dict):
        return False
    return not isinstance(v, np.ndarray)


def _render_inline(d: dict) -> str:
    """Render a dict as 'k1=v1 k2=v2' on a single line."""
    return " ".join(f"{k}={v!r}" for k, v in d.items())


def _summarize_ndarray(
    arr: np.ndarray,
    *,
    include_preview: bool = False,
    full_threshold: int = 20,
    preview_n: int = 5,
) -> dict[str, object]:
    """Compact dict representation of a numpy array.

    Always includes shape and dtype. Includes data only when ``include_preview``
    is True: full contents for small arrays, a head slice for larger ones.
    """
    out = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if include_preview:
        if arr.size <= full_threshold:
            out["data"] = arr.tolist()
        else:
            out["preview"] = arr.flat[:preview_n].tolist()
            out["truncated"] = int(arr.size - preview_n)
    return out


def _format_scalar(x, precision: int = CONFIG["LOG_FANCY_PRECISION"]) -> str:
    """Format a single numeric value for matrix/vector display."""
    if isinstance(x, (complex, np.complexfloating)) and getattr(x, "imag", 0) != 0:
        return f"{x.real:+.{precision}f}{x.imag:+.{precision}f}j"
    if isinstance(x, (complex, np.complexfloating)):
        return f"{x.real:+.{precision}f}"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    return f"{float(x):+.{precision}f}"


def _render_matrix(arr: np.ndarray, indent: str) -> str:
    """Render a 2D ndarray as an aligned ASCII matrix with box borders."""
    rows, cols = arr.shape
    show_rows = min(rows, CONFIG["LOG_FANCY_MAX_ROWS"])
    show_cols = min(cols, CONFIG["LOG_FANCY_MAX_COLS"])

    cells = [[_format_scalar(arr[r, c]) for c in range(show_cols)] for r in range(show_rows)]
    col_widths = [max(len(cells[r][c]) for r in range(show_rows)) for c in range(show_cols)]

    body = []
    for r in range(show_rows):
        row_str = "  ".join(cells[r][c].rjust(col_widths[c]) for c in range(show_cols))
        if cols > show_cols:
            row_str += "  …"
        body.append(f"{indent}│ {row_str} │")
    if rows > show_rows:
        interior = sum(col_widths) + 2 * (show_cols - 1)
        body.append(f"{indent}│ {'…':^{interior}} │")

    interior_width = len(body[0]) - len(indent) - 2
    top = f"{indent}┌{'─' * interior_width}┐"
    bot = f"{indent}└{'─' * interior_width}┘"
    return "\n".join([top, *body, bot])


def _render_vector(arr: np.ndarray, indent: str) -> str:
    """Render a 1D ndarray as a labeled list."""
    if arr.size <= 12:
        cells = [_format_scalar(x) for x in arr]
        width = max(len(c) for c in cells)
        return "\n".join(f"{indent}[{i}] {cells[i].rjust(width)}" for i in range(arr.size))
    head = ", ".join(_format_scalar(x) for x in arr[:5])
    return f"{indent}[{head}, ...] (+{arr.size - 5} more)"


# ---------- encoders & renderers ----------
class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and other common non-serializable types.

    By default, ndarrays are summarized to shape/dtype only. Pass
    ``include_preview=True`` to also include data previews.
    """

    def __init__(self, *args, include_preview: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._include_preview = include_preview

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return _summarize_ndarray(obj, include_preview=self._include_preview)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


def _render_human(
    d: dict,
    indent: int = 0,
    *,
    include_preview: bool = False,
    fancy: bool = False,
) -> str:
    """Indented human-readable rendering of a dict, ndarray-aware.

    When ``fancy=True``, 1D and 2D ndarrays are rendered as aligned vectors
    and box-bordered matrices. Otherwise compact preview form is used.
    """
    pad = "  " * indent
    parts = []
    for k, v in d.items():
        if isinstance(v, dict):
            if len(v) <= CONFIG["LOG_INLINE_MAX_KEYS"] and all(_is_simple(x) for x in v.values()):
                inline = _render_inline(v)
                line = f"{pad}{k}: {inline}"
                if len(line) <= CONFIG["LOG_INLINE_MAX_LEN"]:
                    parts.append(line)
                    continue
            parts.append(f"{pad}{k}:")
            parts.append(
                _render_human(
                    v,
                    indent + 1,
                    include_preview=include_preview,
                    fancy=fancy,
                )
            )
        elif isinstance(v, np.ndarray):
            header = f"{pad}{k}: shape={v.shape} dtype={v.dtype}"
            if not include_preview:
                parts.append(header)
            elif fancy and v.ndim == 2 and v.size > 0:
                parts.append(header)
                parts.append(_render_matrix(v, pad + "  "))
            elif fancy and v.ndim == 1 and v.size > 0:
                parts.append(header)
                parts.append(_render_vector(v, pad + "  "))
            else:
                if v.size <= 20:
                    parts.append(f"{header} {v.tolist()}")
                else:
                    head = v.flat[:5].tolist()
                    parts.append(f"{header} {head}... (+{v.size - 5} more)")
        else:
            parts.append(f"{pad}{k}: {v!r}")
    return "\n".join(parts)


# ---------- formatters ----------

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


def _extract_extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class HumanFormatter(logging.Formatter):
    """Formatter that appends `extra` fields as an indented block.

    Parameters
    ----------
    fmt : str
        Standard logging format string, passed to ``logging.Formatter``.
    fancy : bool, default False
        If True, render 1D and 2D ndarrays as aligned vectors and matrices
        when the record is at DEBUG level. Plain text otherwise.
    """

    def __init__(self, fmt: str | None = None, *, fancy: bool = False):
        super().__init__(fmt)
        self._fancy = fancy

    def format(self, record):
        base = super().format(record)
        extras = _extract_extras(record)
        if not extras:
            return base
        include_preview = record.levelno <= logging.DEBUG
        rendered = _render_human(
            extras,
            include_preview=include_preview,
            fancy=self._fancy and include_preview,
        )
        indented = "\n".join("  " + line for line in rendered.split("\n"))
        return f"{base}\n{indented}"


class JSONFormatter(logging.Formatter):
    """Emits each record as a single JSON line, with `extra` fields merged in.

    ndarray data previews are included only at DEBUG level; INFO and above
    keep just shape/dtype to avoid bloated production logs.
    """

    def __init__(self, encoder_cls: type[json.JSONEncoder] = SafeJSONEncoder):
        super().__init__()
        self._encoder_cls = encoder_cls

    def format(self, record):
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_extract_extras(record),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        include_preview = record.levelno <= logging.DEBUG
        return json.dumps(
            payload,
            cls=self._encoder_cls,
            default=str,
            # forwarded to SafeJSONEncoder.__init__
            **({"include_preview": True} if include_preview else {}),
        )


# ---------- registry ----------

FORMATTERS = {
    "simple": HumanFormatter("%(message)s"),
    "compact": HumanFormatter("[%(levelname)s] %(message)s"),
    "standard": HumanFormatter("%(name)s - [%(levelname)s] %(message)s"),
    "detailed": HumanFormatter(
        "%(asctime)s - %(name)s - [%(levelname)s] "
        "(%(filename)s:%(funcName)s:%(lineno)d) %(message)s"
    ),
    "fancy": HumanFormatter("%(message)s", fancy=True),
    "json": JSONFormatter(),
}


def get_formatter(format_type: str = "simple") -> logging.Formatter:
    if format_type not in FORMATTERS:
        raise ValueError(f"Unknown format: {format_type}. Available: {list(FORMATTERS.keys())}")
    return FORMATTERS[format_type]
