"""AST node definitions for Sprout.

These are plain data classes — no behavior. The parser builds a tree of these,
and the interpreter walks it. Keeping them dumb (no `eval` method on the node)
means evaluation logic lives in one place (interpreter.py), which is easier to
follow than the alternative "each node evaluates itself" style.
"""

from dataclasses import dataclass
from typing import Any, List, Optional


# --- Base classes ---------------------------------------------------------

class Node:
    """Base class for every AST node."""


class Expr(Node):
    """Base class for expressions (things that produce a value)."""


class Stmt(Node):
    """Base class for statements (things that perform an action)."""


# --- Expressions ----------------------------------------------------------

@dataclass
class Literal(Expr):
    """A literal value: number, string, boolean, or nil."""
    value: Any


@dataclass
class Identifier(Expr):
    """A variable reference, e.g. `x`."""
    name: str


@dataclass
class Unary(Expr):
    """A prefix operation, e.g. `-x` or `!flag`."""
    op: str
    operand: Expr


@dataclass
class Binary(Expr):
    """An infix operation, e.g. `a + b` or `a < b`."""
    op: str
    left: Expr
    right: Expr


@dataclass
class Logical(Expr):
    """Short-circuiting `and` / `or` (kept separate from Binary because
    evaluation order matters — the right side may never run)."""
    op: str
    left: Expr
    right: Expr


@dataclass
class Assign(Expr):
    """Assignment to an existing variable, e.g. `x = 5`."""
    name: str
    value: Expr


@dataclass
class Call(Expr):
    """A function call, e.g. `f(a, b)`."""
    callee: Expr
    args: List[Expr]


@dataclass
class ArrayLiteral(Expr):
    """An array literal, e.g. `[1, 2, 3]`."""
    elements: List[Expr]


@dataclass
class MapLiteral(Expr):
    """A hash-map literal, e.g. `{"a": 1, "b": 2}`. `pairs` is a list of
    (key_expr, value_expr) tuples, evaluated in order."""
    pairs: List[tuple]


@dataclass
class Index(Expr):
    """An element read: `arr[i]` for arrays/strings or `map[key]` for maps."""
    target: Expr
    index: Expr


@dataclass
class IndexAssign(Expr):
    """An element write, e.g. `arr[i] = x`. Kept separate from Assign because
    the target is an arbitrary expression plus an index, not a bare name."""
    target: Expr
    index: Expr
    value: Expr


# --- Statements -----------------------------------------------------------

@dataclass
class ExpressionStmt(Stmt):
    """An expression used as a statement, e.g. a bare function call."""
    expr: Expr


@dataclass
class LetStmt(Stmt):
    """Variable declaration, e.g. `let x = 1;`."""
    name: str
    initializer: Optional[Expr]


@dataclass
class Block(Stmt):
    """A `{ ... }` block introducing a new scope."""
    statements: List[Stmt]


@dataclass
class IfStmt(Stmt):
    """`if (cond) { ... } else { ... }`."""
    condition: Expr
    then_branch: Stmt
    else_branch: Optional[Stmt]


@dataclass
class WhileStmt(Stmt):
    """`while (cond) { ... }`."""
    condition: Expr
    body: Stmt


@dataclass
class FunctionStmt(Stmt):
    """Function declaration, e.g. `fn add(a, b) { ... }`."""
    name: str
    params: List[str]
    body: List[Stmt]


@dataclass
class ReturnStmt(Stmt):
    """`return expr;`."""
    value: Optional[Expr]


@dataclass
class PrintStmt(Stmt):
    """`print expr;` — kept as a built-in statement to keep early bootstrapping
    simple (no need for a working function-call path just to see output)."""
    expr: Expr
