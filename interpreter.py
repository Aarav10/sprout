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
        """Bind a new variable in this scope (redefinition is allowed)."""
        self.values[name] = value

    def get(self, name: str) -> Any:
        """Look up a variable, searching enclosing scopes outward."""
        if name in self.values:
            return self.values[name]
        if self.enclosing is not None:
            return self.enclosing.get(name)
        raise RuntimeError_(f"undefined variable '{name}'")

    def assign(self, name: str, value: Any) -> None:
        """Reassign an existing variable, searching enclosing scopes outward."""
        if name in self.values:
            self.values[name] = value
            return
        if self.enclosing is not None:
            self.enclosing.assign(name, value)
            return
        raise RuntimeError_(f"undefined variable '{name}'")


class Function:
    """A callable Sprout function: its declaration plus the environment it was
    defined in (closures)."""

    def __init__(self, declaration: ast.FunctionStmt, closure: Environment):
        self.declaration = declaration
        self.closure = closure

    def call(self, interpreter: "Interpreter", args: List[Any]) -> Any:
        # Each call gets a fresh scope whose parent is the closure, so the
        # function sees the variables in scope where it was *defined*.
        env = Environment(self.closure)
        for param, arg in zip(self.declaration.params, args):
            env.define(param, arg)
        try:
            interpreter.execute_block(self.declaration.body, env)
        except Return as r:
            return r.value
        return None  # functions with no `return` evaluate to nil

    def __repr__(self) -> str:
        return f"<fn {self.declaration.name}>"


class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals

    def interpret(self, statements: List[ast.Stmt]) -> None:
        """Run a whole program (list of top-level statements)."""
        for stmt in statements:
            self.execute(stmt)

    # --- statement execution ---------------------------------------------

    def execute(self, stmt: ast.Stmt) -> None:
        """Dispatch a single statement to its handler."""
        if isinstance(stmt, ast.ExpressionStmt):
            self.evaluate(stmt.expr)
        elif isinstance(stmt, ast.LetStmt):
            value = self.evaluate(stmt.initializer) if stmt.initializer is not None else None
            self.environment.define(stmt.name, value)
        elif isinstance(stmt, ast.Block):
            self.execute_block(stmt.statements, Environment(self.environment))
        elif isinstance(stmt, ast.IfStmt):
            if self._is_truthy(self.evaluate(stmt.condition)):
                self.execute(stmt.then_branch)
            elif stmt.else_branch is not None:
                self.execute(stmt.else_branch)
        elif isinstance(stmt, ast.WhileStmt):
            while self._is_truthy(self.evaluate(stmt.condition)):
                self.execute(stmt.body)
        elif isinstance(stmt, ast.FunctionStmt):
            self.environment.define(stmt.name, Function(stmt, self.environment))
        elif isinstance(stmt, ast.ReturnStmt):
            value = self.evaluate(stmt.value) if stmt.value is not None else None
            raise Return(value)
        elif isinstance(stmt, ast.PrintStmt):
            print(self._stringify(self.evaluate(stmt.expr)))
        else:
            raise RuntimeError_(f"unknown statement type {type(stmt).__name__}")

    def execute_block(self, statements: List[ast.Stmt], env: Environment) -> None:
        """Run a list of statements in `env`, restoring the previous scope after."""
        previous = self.environment
        self.environment = env
        try:
            for stmt in statements:
                self.execute(stmt)
        finally:
            self.environment = previous

    # --- expression evaluation -------------------------------------------

    def evaluate(self, expr: ast.Expr) -> Any:
        """Dispatch a single expression to its handler and return its value."""
        if isinstance(expr, ast.Literal):
            return expr.value
        if isinstance(expr, ast.Identifier):
            return self.environment.get(expr.name)
        if isinstance(expr, ast.Assign):
            value = self.evaluate(expr.value)
            self.environment.assign(expr.name, value)
            return value
        if isinstance(expr, ast.Unary):
            return self._eval_unary(expr)
        if isinstance(expr, ast.Binary):
            return self._eval_binary(expr)
        if isinstance(expr, ast.Logical):
            return self._eval_logical(expr)
        if isinstance(expr, ast.Call):
            return self._eval_call(expr)
        raise RuntimeError_(f"unknown expression type {type(expr).__name__}")

    def _eval_unary(self, expr: ast.Unary) -> Any:
        operand = self.evaluate(expr.operand)
        if expr.op == "-":
            self._check_number(operand, "unary '-'")
            return -operand
        if expr.op == "!":
            return not self._is_truthy(operand)
        raise RuntimeError_(f"unknown unary operator '{expr.op}'")

    def _eval_binary(self, expr: ast.Binary) -> Any:
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        op = expr.op

        # Equality works on any pair of values.
        if op == "==":
            return left == right
        if op == "!=":
            return left != right

        # '+' is overloaded: numeric addition or string concatenation.
        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if self._is_number(left) and self._is_number(right):
                return left + right
            raise RuntimeError_("operands to '+' must be two numbers or two strings")

        # All remaining operators require numbers.
        self._check_number(left, op)
        self._check_number(right, op)
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            if right == 0:
                raise RuntimeError_("division by zero")
            return left / right
        if op == "%":
            if right == 0:
                raise RuntimeError_("modulo by zero")
            return left % right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        raise RuntimeError_(f"unknown binary operator '{op}'")

    def _eval_logical(self, expr: ast.Logical) -> Any:
        left = self.evaluate(expr.left)
        # Short-circuit: return early without evaluating the right side.
        if expr.op == "or":
            if self._is_truthy(left):
                return left
        else:  # "and"
            if not self._is_truthy(left):
                return left
        return self.evaluate(expr.right)

    def _eval_call(self, expr: ast.Call) -> Any:
        callee = self.evaluate(expr.callee)
        args = [self.evaluate(arg) for arg in expr.args]
        if not isinstance(callee, Function):
            raise RuntimeError_("can only call functions")
        expected = len(callee.declaration.params)
        if len(args) != expected:
            raise RuntimeError_(
                f"function '{callee.declaration.name}' expected {expected} "
                f"argument(s) but got {len(args)}"
            )
        return callee.call(self, args)

    # --- helpers ----------------------------------------------------------

    def _is_number(self, value: Any) -> bool:
        # bool is a subclass of int in Python; exclude it so `true * 2` errors.
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _check_number(self, value: Any, op: str) -> None:
        if not self._is_number(value):
            raise RuntimeError_(f"operand to '{op}' must be a number")

    def _is_truthy(self, value: Any) -> bool:
        """Sprout truthiness: nil and false are falsy, everything else truthy."""
        if value is None or value is False:
            return False
        return True

    def _stringify(self, value: Any) -> str:
        """Render a runtime value the way `print` should show it."""
        if value is None:
            return "nil"
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)
