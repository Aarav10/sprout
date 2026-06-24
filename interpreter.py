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

    @property
    def name(self) -> str:
        return self.declaration.name

    def arity(self) -> int:
        return len(self.declaration.params)

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


class NativeFunction:
    """A builtin implemented in Python (e.g. `len`). Same call interface as
    Function so `_eval_call` can treat them uniformly."""

    def __init__(self, name: str, arity: int, fn):
        self.name = name
        self._arity = arity
        self.fn = fn

    def arity(self) -> int:
        return self._arity

    def call(self, interpreter: "Interpreter", args: List[Any]) -> Any:
        return self.fn(args)

    def __repr__(self) -> str:
        return f"<native fn {self.name}>"


class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self._install_builtins()

    def _install_builtins(self) -> None:
        """Register native functions into the global scope."""
        self.globals.define("len", NativeFunction("len", 1, self._builtin_len))
        self.globals.define("push", NativeFunction("push", 2, self._builtin_push))
        self.globals.define("pop", NativeFunction("pop", 1, self._builtin_pop))
        self.globals.define("keys", NativeFunction("keys", 1, self._builtin_keys))
        self.globals.define("has", NativeFunction("has", 2, self._builtin_has))

    # --- builtins ---------------------------------------------------------

    def _builtin_len(self, args: List[Any]) -> Any:
        value = args[0]
        if isinstance(value, (list, str, dict)):
            return len(value)
        raise RuntimeError_("len() expects an array, string, or map")

    def _builtin_keys(self, args: List[Any]) -> Any:
        m = args[0]
        if not isinstance(m, dict):
            raise RuntimeError_("keys() expects a map")
        return list(m.keys())

    def _builtin_has(self, args: List[Any]) -> Any:
        m, key = args
        if not isinstance(m, dict):
            raise RuntimeError_("has() expects a map as its first argument")
        self._check_hashable(key)
        return key in m

    def _builtin_push(self, args: List[Any]) -> Any:
        arr, value = args
        if not isinstance(arr, list):
            raise RuntimeError_("push() expects an array as its first argument")
        arr.append(value)
        return arr  # mutated in place, returned for convenience

    def _builtin_pop(self, args: List[Any]) -> Any:
        arr = args[0]
        if not isinstance(arr, list):
            raise RuntimeError_("pop() expects an array")
        if not arr:
            raise RuntimeError_("pop() from an empty array")
        return arr.pop()

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
        if isinstance(expr, ast.ArrayLiteral):
            return [self.evaluate(element) for element in expr.elements]
        if isinstance(expr, ast.MapLiteral):
            return self._eval_map(expr)
        if isinstance(expr, ast.Index):
            return self._eval_index(expr)
        if isinstance(expr, ast.IndexAssign):
            return self._eval_index_assign(expr)
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
        if not isinstance(callee, (Function, NativeFunction)):
            raise RuntimeError_("can only call functions")
        expected = callee.arity()
        if len(args) != expected:
            raise RuntimeError_(
                f"function '{callee.name}' expected {expected} "
                f"argument(s) but got {len(args)}"
            )
        return callee.call(self, args)

    def _eval_map(self, expr: ast.MapLiteral) -> Any:
        result = {}
        for key_expr, value_expr in expr.pairs:
            key = self.evaluate(key_expr)
            self._check_hashable(key)
            result[key] = self.evaluate(value_expr)
        return result

    def _eval_index(self, expr: ast.Index) -> Any:
        target = self.evaluate(expr.target)
        index = self.evaluate(expr.index)
        if isinstance(target, dict):
            self._check_hashable(index)  # avoid a raw TypeError from `in`
            if index not in target:
                raise RuntimeError_(
                    f"key {self._stringify_element(index)} not found in map"
                )
            return target[index]
        if isinstance(target, (list, str)):
            i = self._resolve_index(target, index)
            return target[i]
        raise RuntimeError_("can only index into arrays, strings, and maps")

    def _eval_index_assign(self, expr: ast.IndexAssign) -> Any:
        target = self.evaluate(expr.target)
        index = self.evaluate(expr.index)
        value = self.evaluate(expr.value)
        if isinstance(target, dict):
            self._check_hashable(index)
            target[index] = value  # inserts or updates; maps grow
            return value
        if isinstance(target, list):
            i = self._resolve_index(target, index)
            target[i] = value
            return value
        raise RuntimeError_("can only assign to array elements or map keys")

    def _check_hashable(self, key: Any) -> None:
        """Maps accept string/number/boolean/nil keys; arrays and maps can't
        be keys (they're mutable / unhashable)."""
        if isinstance(key, (list, dict)):
            raise RuntimeError_("map key must be a string, number, boolean, or nil")

    def _resolve_index(self, seq: Any, index: Any) -> int:
        """Validate an index and normalize negatives (Python-style), or raise."""
        if not isinstance(index, int) or isinstance(index, bool):
            raise RuntimeError_("index must be an integer")
        n = len(seq)
        i = index + n if index < 0 else index
        if i < 0 or i >= n:
            raise RuntimeError_(f"index {index} out of range for length {n}")
        return i

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
        if isinstance(value, list):
            return "[" + ", ".join(self._stringify_element(e) for e in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(
                f"{self._stringify_element(k)}: {self._stringify_element(v)}"
                for k, v in value.items()
            )
            return "{" + items + "}"
        return str(value)

    def _stringify_element(self, value: Any) -> str:
        """Like _stringify but quotes strings, so `[1, "a"]` is unambiguous."""
        if isinstance(value, str):
            return '"' + value + '"'
        return self._stringify(value)
