"""One place to decide a report's format and write it (#269 item 1.3).

Sixteen CLIs carried their own copy of the same eight-line
``_detect_format``, and the copies had already drifted: most mapped
``.markdown`` to ``md``, two did not. A CLI's ``--output`` contract is part
of the repo's conventions ("all CLI tools support ``--output`` with
auto-detected format"), so it belongs in one function rather than in
sixteen.

    from graphs.reporting.output_format import detect_format, write_report

    fmt = detect_format(args.output)          # "text" | "json" | "csv" | "md"
    ...
    write_report(payload, args.output)        # file, or stdout when None

``text`` is the default for no ``--output`` and for any unrecognized
extension: a report is printed rather than refused.
"""

from __future__ import annotations

import os
import sys
from typing import IO, Optional, Tuple

#: The formats every CLI here can emit.
FORMATS: Tuple[str, ...] = ("text", "json", "csv", "md")

#: Filename extension (lowercase, no dot) -> format.
EXTENSION_FORMATS = {
    "json": "json",
    "csv": "csv",
    "md": "md",
    "markdown": "md",
    "txt": "text",
    "text": "text",
}


def detect_format(output: Optional[str], force: Optional[str] = None) -> str:
    """The format implied by an ``--output`` path.

    ``None`` (no ``--output``) and unrecognized extensions both give
    ``"text"``, so a CLI writing ``report.out`` gets a readable file instead
    of an error.

    ``force`` is for a CLI that also has an explicit flag -- ``show_floorplan
    --json`` -- and wins over the extension. It is normalized through
    ``EXTENSION_FORMATS`` so ``"markdown"`` works there too.
    """
    if force:
        return EXTENSION_FORMATS.get(str(force).lower().lstrip("."), str(force).lower())
    if not output:
        return "text"
    ext = os.path.splitext(str(output))[1].lower().lstrip(".")
    return EXTENSION_FORMATS.get(ext, "text")


def write_report(
    payload: str,
    output: Optional[str] = None,
    *,
    announce: bool = True,
    stream: Optional[IO[str]] = None,
) -> None:
    """Write ``payload`` to ``output``, or print it when there is no path.

    ``announce`` prints "wrote <path>" to stdout after writing a file, which
    is what the existing CLIs do; pass ``False`` to stay silent. ``stream``
    overrides where the payload goes when there is no ``output`` path.
    """
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(payload)
        if announce:
            print(f"wrote {output}")
        return
    print(payload, end="" if payload.endswith("\n") else "\n", file=stream or sys.stdout)
