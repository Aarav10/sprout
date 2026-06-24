"""Hand-written recursive-descent parser for Sprout.

Consumes the token list from the lexer and produces an AST (ast_nodes). One
method per grammar rule. Precedence is encoded by the call chain: lower-
precedence rules call higher-precedence ones.

Grammar (informal, lowest to highest precedence):

    program     -> statement* EOF
    statement   -> letDecl | fnDecl | ifStmt | whileStmt
                 | returnStmt | printStmt | block | exprStmt
    block       -> "{" statement* "}"
    exprStmt    -> expression ";"

    expression  -> assignment
    assignment  -> IDENT "=" assignment | logic_or
    logic_or    -> logic_and ( "or" logic_and )*
    logic_and   -> equality ( "and" equality )*
    equality    -> comparison ( ("==" | "!=") comparison )*
    comparison  -> term ( ("<" | "<=" | ">" | ">=") term )*
    term        -> factor ( ("+" | "-") factor )*
    factor      -> unary ( ("*" | "/" | "%") unary )*
    unary       -> ("!" | "-") unary | call
    call        -> primary ( "(" arguments? ")" )*
    primary     -> NUMBER | STRING | "true" | "false" | "nil"
                 | IDENT | "(" expression ")"
"""

from typing import List, Optional

import ast_nodes as ast
from lexer import Token, TokenType


class ParseError(Exception):
    """Raised when the token stream doesn't match the grammar."""


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse(self) -> List[ast.Stmt]:
        """Parse the whole token stream into a list of top-level statements."""
        raise NotImplementedError

    # --- statement rules --------------------------------------------------

    def _statement(self) -> ast.Stmt:
        raise NotImplementedError

    def _let_declaration(self) -> ast.Stmt:
        raise NotImplementedError

    def _fn_declaration(self) -> ast.Stmt:
        raise NotImplementedError

    def _if_statement(self) -> ast.Stmt:
        raise NotImplementedError

    def _while_statement(self) -> ast.Stmt:
        raise NotImplementedError

    def _return_statement(self) -> ast.Stmt:
        raise NotImplementedError

    def _print_statement(self) -> ast.Stmt:
        raise NotImplementedError

    def _block(self) -> List[ast.Stmt]:
        raise NotImplementedError

    def _expression_statement(self) -> ast.Stmt:
        raise NotImplementedError

    # --- expression rules (precedence climbing) ---------------------------

    def _expression(self) -> ast.Expr:
        raise NotImplementedError

    def _assignment(self) -> ast.Expr:
        raise NotImplementedError

    def _logic_or(self) -> ast.Expr:
        raise NotImplementedError

    def _logic_and(self) -> ast.Expr:
        raise NotImplementedError

    def _equality(self) -> ast.Expr:
        raise NotImplementedError

    def _comparison(self) -> ast.Expr:
        raise NotImplementedError

    def _term(self) -> ast.Expr:
        raise NotImplementedError

    def _factor(self) -> ast.Expr:
        raise NotImplementedError

    def _unary(self) -> ast.Expr:
        raise NotImplementedError

    def _call(self) -> ast.Expr:
        raise NotImplementedError

    def _primary(self) -> ast.Expr:
        raise NotImplementedError

    # --- token cursor helpers ---------------------------------------------

    def _match(self, *types: TokenType) -> bool:
        """Consume and return True if the current token is one of `types`."""
        raise NotImplementedError

    def _check(self, type: TokenType) -> bool:
        """True if the current token is `type` (without consuming)."""
        raise NotImplementedError

    def _advance(self) -> Token:
        """Consume and return the current token."""
        raise NotImplementedError

    def _consume(self, type: TokenType, message: str) -> Token:
        """Consume the expected token or raise ParseError(message)."""
        raise NotImplementedError

    def _peek(self) -> Token:
        raise NotImplementedError

    def _previous(self) -> Token:
        raise NotImplementedError

    def _is_at_end(self) -> bool:
        raise NotImplementedError
