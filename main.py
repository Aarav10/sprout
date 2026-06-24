"""Entry point: run a Sprout source file, or start an interactive REPL.

Usage:
    python main.py path/to/program.txt   # run a file
    python main.py                       # start the REPL

Pipeline:  source text -> Lexer -> tokens -> Parser -> AST -> Interpreter
"""

import sys

import ast_nodes as ast
from lexer import Lexer, LexError
from parser import Parser, ParseError
from interpreter import Interpreter, RuntimeError_


def run(source: str) -> int:
    """Run a program from source text. Returns a process exit code."""
    try:
        tokens = Lexer(source).tokenize()
        statements = Parser(tokens).parse()
        Interpreter().interpret(statements)
    except (LexError, ParseError) as err:
        print(f"Syntax error: {err}", file=sys.stderr)
        return 65
    except RuntimeError_ as err:
        print(f"Runtime error: {err}", file=sys.stderr)
        return 70
    return 0


def run_repl() -> int:
    """Start an interactive read-eval-print loop.

    One Interpreter is reused for the whole session, so variables and
    functions defined on one line stay available on later lines.
    """
    interpreter = Interpreter()
    print("Sprout REPL — type 'exit' (or Ctrl-D) to quit.")
    while True:
        try:
            line = input(">>> ")
        except EOFError:        # Ctrl-D
            print()
            break
        except KeyboardInterrupt:  # Ctrl-C cancels the current line, not the session
            print("\n(interrupted)")
            continue

        stripped = line.strip()
        if stripped == "":
            continue
        if stripped in ("exit", "quit"):
            break

        _eval_repl_line(interpreter, line)
    return 0


def _eval_repl_line(interpreter: Interpreter, source: str) -> None:
    """Evaluate one line of REPL input against the shared interpreter.

    Convenience: a bare expression without a trailing ';' is still accepted,
    and its value is echoed back (so `1 + 2` prints `3`). Errors are printed
    but never end the session.
    """
    try:
        try:
            statements = Parser(Lexer(source).tokenize()).parse()
        except (LexError, ParseError):
            # Retry treating the input as a bare expression statement.
            statements = Parser(Lexer(source + ";").tokenize()).parse()

        # Echo the result of a single bare expression (REPL ergonomics).
        if len(statements) == 1 and isinstance(statements[0], ast.ExpressionStmt):
            value = interpreter.evaluate(statements[0].expr)
            if value is not None:
                print(interpreter._stringify(value))
        else:
            interpreter.interpret(statements)
    except (LexError, ParseError) as err:
        print(f"Syntax error: {err}", file=sys.stderr)
    except RuntimeError_ as err:
        print(f"Runtime error: {err}", file=sys.stderr)


def main(argv: list) -> int:
    if len(argv) == 1:
        return run_repl()
    if len(argv) != 2:
        print("usage: python main.py [source.txt]", file=sys.stderr)
        return 64
    path = argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as err:
        print(f"Could not read {path}: {err}", file=sys.stderr)
        return 66
    return run(source)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
