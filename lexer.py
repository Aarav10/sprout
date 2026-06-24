"""Hand-written lexer (tokenizer) for Sprout.

Turns raw source text into a flat list of Tokens. The parser consumes that
list. Everything here is character-by-character scanning — no regex engine
driving the structure, no external lib.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List

from errors import SproutError


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    ISTRING = auto()       # interpolated string: literal is a list of parts
    IDENT = auto()

    # Keywords
    LET = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
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
    COLON = auto()
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
    "for": TokenType.FOR,
    "return": TokenType.RETURN,
    "print": TokenType.PRINT,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "nil": TokenType.NIL,
    "and": TokenType.AND,
    "or": TokenType.OR,
}

# Backslash escape sequences recognized inside string literals. Anything else
# after a backslash is a lex error.
STRING_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "0": "\0",
    "\\": "\\",
    '"': '"',
    "{": "{",   # escape a literal brace so it isn't read as interpolation
    "}": "}",
}


@dataclass
class Token:
    type: TokenType
    lexeme: str          # the exact source text
    literal: Any         # parsed value for NUMBER/STRING, else None
    line: int            # 1-based line number, for error messages
    column: int = 1      # 1-based column of the token's first character


class LexError(SproutError):
    """Raised on an unexpected character or unterminated string."""


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0       # start of the token currently being scanned
        self.current = 0     # current scan position
        self.line = 1
        self.line_start = 0  # source index where the current line begins

    def tokenize(self) -> List[Token]:
        """Scan the whole source and return the token list (ending in EOF)."""
        while not self._is_at_end():
            self.start = self.current
            self._scan_token()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line, self._column(self.current)))
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
            self.line_start = self.current  # next char starts the new line
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
            ":": TokenType.COLON,
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

        raise LexError(f"unexpected character {c!r}", self.line, self._column(self.start))

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
        """Scan a double-quoted string literal, translating escape sequences.

        A `\\n` in the source becomes a real newline in the value. A `{expr}`
        starts an interpolation: the string is emitted as an ISTRING token
        whose literal is a list of ("lit", text) / ("expr", source) parts. A
        plain string with no interpolation stays a STRING token.
        """
        start_line = self.line
        start_col = self._column(self.start)
        parts = []           # accumulated ("lit"|"expr", value) parts
        chars: List[str] = []  # current literal run
        interpolated = False

        while self._peek() != '"' and not self._is_at_end():
            c = self._advance()
            if c == "\\":
                # A backslash at end-of-input falls through to the
                # unterminated-string check below.
                if self._is_at_end():
                    break
                esc = self._advance()
                if esc not in STRING_ESCAPES:
                    raise LexError(
                        f"unknown escape sequence '\\{esc}'",
                        self.line, self._column(self.current - 2),
                    )
                chars.append(STRING_ESCAPES[esc])
            elif c == "{":
                interpolated = True
                parts.append(("lit", "".join(chars)))
                chars = []
                parts.append(("expr", self._scan_interpolation()))
            else:
                if c == "\n":
                    self.line += 1
                    self.line_start = self.current
                chars.append(c)

        if self._is_at_end():
            raise LexError("unterminated string", start_line, start_col)

        self._advance()  # consume the closing quote

        if not interpolated:
            self._add_token(TokenType.STRING, "".join(chars))
        else:
            parts.append(("lit", "".join(chars)))
            self._add_token(TokenType.ISTRING, parts)

    def _scan_interpolation(self) -> str:
        """Collect the raw source of a `{...}` expression (the '{' is already
        consumed), up to the matching '}'. Tracks brace depth and skips nested
        string literals so braces inside them don't end the interpolation."""
        start = self.current
        depth = 1
        while not self._is_at_end():
            c = self._peek()
            if c == '"':
                self._skip_nested_string()
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    source = self.source[start:self.current]
                    self._advance()  # consume the '}'
                    if source.strip() == "":
                        raise LexError("empty interpolation '{}'",
                                       self.line, self._column(start))
                    return source
            elif c == "\n":
                self.line += 1
                self._advance()
                self.line_start = self.current
                continue
            self._advance()
        raise LexError("unterminated interpolation in string",
                       self.line, self._column(start))

    def _skip_nested_string(self) -> None:
        """Advance past a double-quoted string nested inside an interpolation,
        honoring backslash escapes. The text is left in place for re-lexing."""
        self._advance()  # opening quote
        while self._peek() != '"' and not self._is_at_end():
            if self._peek() == "\\":
                self._advance()  # skip the escaped char as a unit
            self._advance()
        if not self._is_at_end():
            self._advance()  # closing quote

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

    def _column(self, index: int) -> int:
        """1-based column of `index` within its source line."""
        return index - self.line_start + 1

    def _add_token(self, type: TokenType, literal: Any = None) -> None:
        lexeme = self.source[self.start:self.current]
        self.tokens.append(Token(type, lexeme, literal, self.line, self._column(self.start)))
