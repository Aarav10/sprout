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
    LBRACKET = auto()
    RBRACKET = auto()
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
        while not self._is_at_end():
            self.start = self.current
            self._scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    # --- helpers ----------------------------------------------------------

    def _scan_token(self) -> None:
        """Scan a single token starting at self.start."""
        c = self._advance()

        # Whitespace: skip. Track newlines for line numbers.
        if c in " \r\t":
            return
        if c == "\n":
            self.line += 1
            return

        # Single-character tokens.
        simple = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "%": TokenType.PERCENT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "{": TokenType.LBRACE,
            "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ",": TokenType.COMMA,
            ";": TokenType.SEMICOLON,
        }
        if c in simple:
            self._add_token(simple[c])
            return

        # Operators that may be one or two characters.
        if c == "=":
            self._add_token(TokenType.EQ if self._match("=") else TokenType.ASSIGN)
            return
        if c == "!":
            self._add_token(TokenType.NEQ if self._match("=") else TokenType.BANG)
            return
        if c == "<":
            self._add_token(TokenType.LTE if self._match("=") else TokenType.LT)
            return
        if c == ">":
            self._add_token(TokenType.GTE if self._match("=") else TokenType.GT)
            return

        # Slash is either a `//` line comment or division.
        if c == "/":
            if self._match("/"):
                # Comment runs to end of line; consume but emit no token.
                while self._peek() != "\n" and not self._is_at_end():
                    self._advance()
            else:
                self._add_token(TokenType.SLASH)
            return

        # Literals and identifiers.
        if c == '"':
            self._string()
            return
        if c.isdigit():
            self._number()
            return
        if c.isalpha() or c == "_":
            self._identifier()
            return

        raise LexError(f"Unexpected character {c!r} on line {self.line}")

    def _number(self) -> None:
        """Scan a numeric literal (integer or float)."""
        while self._peek().isdigit():
            self._advance()

        # Look for a fractional part: a '.' followed by at least one digit.
        is_float = False
        if self._peek() == "." and self._peek_next().isdigit():
            is_float = True
            self._advance()  # consume the '.'
            while self._peek().isdigit():
                self._advance()

        text = self.source[self.start:self.current]
        value = float(text) if is_float else int(text)
        self._add_token(TokenType.NUMBER, value)

    def _string(self) -> None:
        """Scan a double-quoted string literal (no escape sequences yet)."""
        while self._peek() != '"' and not self._is_at_end():
            if self._peek() == "\n":
                self.line += 1
            self._advance()

        if self._is_at_end():
            raise LexError(f"Unterminated string starting on line {self.line}")

        self._advance()  # consume the closing quote
        # Strip the surrounding quotes for the literal value.
        value = self.source[self.start + 1:self.current - 1]
        self._add_token(TokenType.STRING, value)

    def _identifier(self) -> None:
        """Scan an identifier, promoting it to a keyword token if it is one."""
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()

        text = self.source[self.start:self.current]
        token_type = KEYWORDS.get(text, TokenType.IDENT)
        self._add_token(token_type)

    # --- low-level cursor helpers -----------------------------------------

    def _is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def _advance(self) -> str:
        """Consume the current character and return it."""
        c = self.source[self.current]
        self.current += 1
        return c

    def _match(self, expected: str) -> bool:
        """Consume the current character only if it equals `expected`."""
        if self._is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        return True

    def _peek(self) -> str:
        """Current character without consuming it ('' at end)."""
        if self._is_at_end():
            return ""
        return self.source[self.current]

    def _peek_next(self) -> str:
        """The character after the current one ('' at end)."""
        if self.current + 1 >= len(self.source):
            return ""
        return self.source[self.current + 1]

    def _add_token(self, type: TokenType, literal: Any = None) -> None:
        lexeme = self.source[self.start:self.current]
        self.tokens.append(Token(type, lexeme, literal, self.line))
