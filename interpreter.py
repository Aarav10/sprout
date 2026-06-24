"""Tree-walking interpreter for Sprout.

Walks the AST produced by the parser and evaluates it directly (no bytecode,
no compilation step). Statements are executed for their effect; expressions
are evaluated to a Python value (we reuse Python's int/float/str/bool/None as
Sprout's runtime values).
"""

from typing import Any, Dict, List, Optional

import ast_nodes as ast


class RuntimeError_(Exception):
    """A Sprout runtime error (named with a trailing underscore so it doesn't
    shadow Python's built-in RuntimeError)."""


class Return(Exception):
    """Control-flow exception used to unwind out of a function body when a
    `return` statement runs. Carries the return value."""
    def __init__(self, value: Any):
        self.value = value


class Environment:
    """A single scope: a name->value map plus a link to the enclosing scope.
    Variable lookup walks outward through `enclosing` until found."""

    def __init__(self, enclosing: Optional["Environment"] = None):
        self.values: Dict[str, Any] = {}
        self.enclosing = enclosing

    def define(self, name: str, value: Any) -> None:
        """Bind a new variable in this scope."""
        raise NotImplementedError

    def get(self, name: str) -> Any:
        """Look up a variable, searching enclosing scopes."""
        raise NotImplementedError

    def assign(self, name: str, value: Any) -> None:
        """Reassign an existing variable, searching enclosing scopes."""
        raise NotImplementedError


class Function:
    """A callable Sprout function: its declaration plus the environment it was
    defined in (closures)."""

    def __init__(self, declaration: ast.FunctionStmt, closure: Environment):
        self.declaration = declaration
        self.closure = closure

    def call(self, interpreter: "Interpreter", args: List[Any]) -> Any:
        raise NotImplementedError


class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals

    def interpret(self, statements: List[ast.Stmt]) -> None:
        """Run a whole program (list of top-level statements)."""
        raise NotImplementedError

    # --- statement execution ---------------------------------------------

    def execute(self, stmt: ast.Stmt) -> None:
        """Dispatch a single statement to its handler."""
        raise NotImplementedError

    def execute_block(self, statements: List[ast.Stmt], env: Environment) -> None:
        """Run a list of statements in a fresh scope."""
        raise NotImplementedError

    # --- expression evaluation -------------------------------------------

    def evaluate(self, expr: ast.Expr) -> Any:
        """Dispatch a single expression to its handler and return its value."""
        raise NotImplementedError

    # --- helpers ----------------------------------------------------------

    def _is_truthy(self, value: Any) -> bool:
        """Sprout truthiness: nil and false are falsy, everything else truthy."""
        raise NotImplementedError

    def _stringify(self, value: Any) -> str:
        """Render a runtime value the way `print` should show it."""
        raise NotImplementedError
