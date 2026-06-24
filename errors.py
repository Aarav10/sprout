"""Shared error type and source-pointing formatter for Sprout.

Every pipeline stage raises a SproutError (or a subclass). Carrying the line
and column lets `format` render the offending source line with a caret under
the problem, e.g.

    Syntax error: unexpected character '@' (line 2)

      2 | let x = @;
        |         ^
"""


class SproutError(Exception):
    """Base for all Sprout errors. `line`/`column` are 1-based; either may be
    None when unknown (the formatter degrades gracefully)."""

    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def format(self, source, kind):
        """Render the error against `source`. `kind` is a human label such as
        'Syntax error' or 'Runtime error'."""
        header = f"{kind}: {self.message}"
        if self.line is None:
            return header

        lines = source.splitlines()
        if not (1 <= self.line <= len(lines)):
            return f"{header} (line {self.line})"

        src_line = lines[self.line - 1]
        if self.column is not None:
            col = self.column
        else:
            # No column: point at the first non-blank character on the line.
            col = len(src_line) - len(src_line.lstrip()) + 1
        col = max(1, min(col, len(src_line) + 1))

        num = str(self.line)
        gutter = f"  {num} | "
        cont = f"  {' ' * len(num)} | "
        pointer = cont + " " * (col - 1) + "^"
        return f"{header} (line {self.line})\n\n{gutter}{src_line}\n{pointer}"
