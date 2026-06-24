"""Entry point: run a Sprout source file.

Usage:
    python main.py path/to/program.txt

Pipeline:  source text -> Lexer -> tokens -> Parser -> AST -> Interpreter
"""

import sys

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


def main(argv: list) -> int:
    if len(argv) != 2:
        print("usage: python main.py <source.txt>", file=sys.stderr)
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
