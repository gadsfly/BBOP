import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

try:
    import pyarrow as pa
except ImportError:
    pa = None  # only needed if you pass a pyarrow.Table

Row = Mapping[str, Any]
BuildFn = Callable[[Row], Dict[str, Any]]  # build kwargs from one row

def _rows_from_table(items: Union["pa.Table", Iterable[Row], Row, str]) -> Iterable[Row]:
    """
    Accepts:
      - pyarrow.Table
      - iterable of dict-like rows
      - a single dict-like row
      - a single path string (treated as {'session_root': path})
    Yields row dicts.
    """
    if pa is not None and isinstance(items, pa.Table):
        cols = items.column_names
        arrays = [items.column(c) for c in cols]
        n = items.num_rows
        for i in range(n):
            yield {c: arrays[j][i].as_py() for j, c in enumerate(cols)}
        return

    if isinstance(items, str):
        yield {"session_root": items}
        return

    if isinstance(items, Mapping):
        yield items  # single row dict
        return

    # iterable of rows
    for r in items:
        yield r


def exec_from_table(
    items: Union["pa.Table", Iterable[Row], Row, str],
    fn: Callable[..., Any],
    *,
    build: Optional[BuildFn] = None,
    colmap: Optional[Dict[str, Union[str, BuildFn]]] = None,
    const: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    on_error: str = "print",   # "print" | "raise" | "skip"
) -> List[Dict[str, Any]]:
    """
    Minimal general executor.

    Args:
      items: pyarrow.Table | iterable of row dicts | single row dict | single path str
      fn:    function to call per row
      build: optional lambda(row)->kwargs (free-form kwarg builder)
      colmap: param_name -> column_name OR lambda(row)->value
      const: constant kwargs merged last (wins)
      dry_run: if True, don't call fn; just return plans
      on_error: "print" (default), "raise", or "skip"

    Returns: list of {status, kwargs, result|error}
    """
    colmap = colmap or {}
    const  = const or {}
    out: List[Dict[str, Any]] = []

    for row in _rows_from_table(items):
        try:
            # base kwargs from build(row)
            kwargs = build(row) if build else {}

            # fill from colmap (column name or lambda)
            for k, spec in colmap.items():
                if isinstance(spec, str):
                    kwargs[k] = row.get(spec)
                else:  # callable
                    kwargs[k] = spec(row)

            # merge constants last
            kwargs.update(const)

            if dry_run:
                out.append({"status": "plan", "kwargs": kwargs})
                continue

            result = fn(**kwargs)
            out.append({"status": "ok", "kwargs": kwargs, "result": result})

        except Exception as e:
            rec = {"status": "fail", "kwargs": locals().get("kwargs", {}), "error": repr(e)}
            out.append(rec)
            if on_error == "raise":
                raise
            elif on_error == "print":
                print(f"[exec_from_table] error: {e}")
            # if "skip": just continue

    return out
