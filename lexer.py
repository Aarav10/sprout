"""Hand-written lexer (tokenizer) for Sprout.

Turns raw source text into a flat list of Tokens. The parser consumes that
list. Everything here is character-by-character scanning — no regex engine
driving the structure, no external lib.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    # Keywords
    LET = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    RETURN = auto()
    PRINT = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    AND = auto()
    OR = auto()

    # Single / double char operators and punctuation
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()        # =
    EQ = auto()            # ==
    BANG = auto()          # !
    NEQ = auto()           # !=
    LT = auto()            # <
    LTE = auto()           # <=
    GT = auto()            # >
    GTE = auto()           # >=
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()

    EOF = auto()


# Reserved words mapped to their token type. Anything not here that looks like
# an identifier becomes TokenType.IDENT.
KEYWORDS = {
    "let": TokenType.LET,
    "fn": TokenType.FN,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "return": TokenType.RETURN,
    "print": TokenType.PRINT,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "nil": TokenType.NIL,
    "and": TokenType.AND,
    "or": TokenType.OR,
}


@dataclass
class Token:
    type: TokenType
    lexeme: str          # the exact source text
    literal: Any         # parsed value for NUMBER/STRING, else None
    line: int            # 1-based line number, for error messages


class LexError(Exception):
    """Raised on an unexpected character or unterminated string."""


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0      # start of the token currently being scanned
        self.current = 0    # current scan position
        self.line = 1

    def tokenize(self) -> List[Token]:
        """Scan the whole source and return the token list (ending in EOF)."""
        raise NotImplementedError

    # --- helpers (to be filled in) ---------------------------------------

    def _scan_token(self) -> None:
        """Scan a single token starting at self.current."""
        raise NotImplementedError

    def _number(self) -> None:
        """Scan a numeric literal."""
        raise NotImplementedError

    def _string(self) -> None:
        """Scan a double-quoted string literal."""
        raise NotImplementedError

    def _identifier(self) -> None:
        """Scan an identifier or keyword."""
        raise NotImplementedError
