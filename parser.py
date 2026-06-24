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
        statements: List[ast.Stmt] = []
        while not self._is_at_end():
            statements.append(self._statement())
        return statements

    # --- statement rules --------------------------------------------------

    def _statement(self) -> ast.Stmt:
        """Dispatch on the leading token to the right statement rule."""
        if self._match(TokenType.LET):
            return self._let_declaration()
        if self._match(TokenType.FN):
            return self._fn_declaration()
        if self._match(TokenType.IF):
            return self._if_statement()
        if self._match(TokenType.WHILE):
            return self._while_statement()
        if self._match(TokenType.RETURN):
            return self._return_statement()
        if self._match(TokenType.PRINT):
            return self._print_statement()
        if self._match(TokenType.LBRACE):
            return ast.Block(self._block())
        return self._expression_statement()

    def _let_declaration(self) -> ast.Stmt:
        # `let` already consumed.
        name = self._consume(TokenType.IDENT, "expected variable name after 'let'")
        initializer: Optional[ast.Expr] = None
        if self._match(TokenType.ASSIGN):
            initializer = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after variable declaration")
        return ast.LetStmt(name.lexeme, initializer)

    def _fn_declaration(self) -> ast.Stmt:
        # `fn` already consumed.
        name = self._consume(TokenType.IDENT, "expected function name after 'fn'")
        self._consume(TokenType.LPAREN, "expected '(' after function name")
        params: List[str] = []
        if not self._check(TokenType.RPAREN):
            while True:
                param = self._consume(TokenType.IDENT, "expected parameter name")
                params.append(param.lexeme)
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "expected ')' after parameters")
        self._consume(TokenType.LBRACE, "expected '{' before function body")
        body = self._block()
        return ast.FunctionStmt(name.lexeme, params, body)

    def _if_statement(self) -> ast.Stmt:
        # `if` already consumed.
        self._consume(TokenType.LPAREN, "expected '(' after 'if'")
        condition = self._expression()
        self._consume(TokenType.RPAREN, "expected ')' after if condition")
        then_branch = self._statement()
        else_branch: Optional[ast.Stmt] = None
        if self._match(TokenType.ELSE):
            else_branch = self._statement()
        return ast.IfStmt(condition, then_branch, else_branch)

    def _while_statement(self) -> ast.Stmt:
        # `while` already consumed.
        self._consume(TokenType.LPAREN, "expected '(' after 'while'")
        condition = self._expression()
        self._consume(TokenType.RPAREN, "expected ')' after while condition")
        body = self._statement()
        return ast.WhileStmt(condition, body)

    def _return_statement(self) -> ast.Stmt:
        # `return` already consumed.
        value: Optional[ast.Expr] = None
        if not self._check(TokenType.SEMICOLON):
            value = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after return value")
        return ast.ReturnStmt(value)

    def _print_statement(self) -> ast.Stmt:
        # `print` already consumed.
        value = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after value")
        return ast.PrintStmt(value)

    def _block(self) -> List[ast.Stmt]:
        # Opening '{' already consumed.
        statements: List[ast.Stmt] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            statements.append(self._statement())
        self._consume(TokenType.RBRACE, "expected '}' after block")
        return statements

    def _expression_statement(self) -> ast.Stmt:
        expr = self._expression()
        self._consume(TokenType.SEMICOLON, "expected ';' after expression")
        return ast.ExpressionStmt(expr)

    # --- expression rules (precedence climbing) ---------------------------

    def _expression(self) -> ast.Expr:
        return self._assignment()

    def _assignment(self) -> ast.Expr:
        # Parse the left side as a normal (higher-precedence) expression, then
        # if an '=' follows, treat what we parsed as an assignment target.
        expr = self._logic_or()
        if self._match(TokenType.ASSIGN):
            equals = self._previous()
            value = self._assignment()  # right-associative
            if isinstance(expr, ast.Identifier):
                return ast.Assign(expr.name, value)
            raise ParseError(f"invalid assignment target on line {equals.line}")
        return expr

    def _logic_or(self) -> ast.Expr:
        expr = self._logic_and()
        while self._match(TokenType.OR):
            op = self._previous().lexeme
            right = self._logic_and()
            expr = ast.Logical(op, expr, right)
        return expr

    def _logic_and(self) -> ast.Expr:
        expr = self._equality()
        while self._match(TokenType.AND):
            op = self._previous().lexeme
            right = self._equality()
            expr = ast.Logical(op, expr, right)
        return expr

    def _equality(self) -> ast.Expr:
        expr = self._comparison()
        while self._match(TokenType.EQ, TokenType.NEQ):
            op = self._previous().lexeme
            right = self._comparison()
            expr = ast.Binary(op, expr, right)
        return expr

    def _comparison(self) -> ast.Expr:
        expr = self._term()
        while self._match(TokenType.LT, TokenType.LTE, TokenType.GT, TokenType.GTE):
            op = self._previous().lexeme
            right = self._term()
            expr = ast.Binary(op, expr, right)
        return expr

    def _term(self) -> ast.Expr:
        expr = self._factor()
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op = self._previous().lexeme
            right = self._factor()
            expr = ast.Binary(op, expr, right)
        return expr

    def _factor(self) -> ast.Expr:
        expr = self._unary()
        while self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self._previous().lexeme
            right = self._unary()
            expr = ast.Binary(op, expr, right)
        return expr

    def _unary(self) -> ast.Expr:
        if self._match(TokenType.BANG, TokenType.MINUS):
            op = self._previous().lexeme
            operand = self._unary()  # allow stacking, e.g. --x or !!x
            return ast.Unary(op, operand)
        return self._call()

    def _call(self) -> ast.Expr:
        expr = self._primary()
        # Loop so chained calls like f(a)(b) parse correctly.
        while self._match(TokenType.LPAREN):
            expr = self._finish_call(expr)
        return expr

    def _finish_call(self, callee: ast.Expr) -> ast.Expr:
        # Opening '(' already consumed.
        args: List[ast.Expr] = []
        if not self._check(TokenType.RPAREN):
            while True:
                args.append(self._expression())
                if not self._match(TokenType.COMMA):
                    break
        self._consume(TokenType.RPAREN, "expected ')' after arguments")
        return ast.Call(callee, args)

    def _primary(self) -> ast.Expr:
        if self._match(TokenType.TRUE):
            return ast.Literal(True)
        if self._match(TokenType.FALSE):
            return ast.Literal(False)
        if self._match(TokenType.NIL):
            return ast.Literal(None)
        if self._match(TokenType.NUMBER, TokenType.STRING):
            return ast.Literal(self._previous().literal)
        if self._match(TokenType.IDENT):
            return ast.Identifier(self._previous().lexeme)
        if self._match(TokenType.LPAREN):
            expr = self._expression()
            self._consume(TokenType.RPAREN, "expected ')' after expression")
            return expr
        tok = self._peek()
        raise ParseError(f"expected expression but found {tok.lexeme!r} on line {tok.line}")

    # --- token cursor helpers ---------------------------------------------

    def _match(self, *types: TokenType) -> bool:
        """Consume and return True if the current token is one of `types`."""
        for type in types:
            if self._check(type):
                self._advance()
                return True
        return False

    def _check(self, type: TokenType) -> bool:
        """True if the current token is `type` (without consuming)."""
        if self._is_at_end():
            return False
        return self._peek().type == type

    def _advance(self) -> Token:
        """Consume and return the current token."""
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _consume(self, type: TokenType, message: str) -> Token:
        """Consume the expected token or raise ParseError(message)."""
        if self._check(type):
            return self._advance()
        tok = self._peek()
        raise ParseError(f"{message} (line {tok.line}, found {tok.lexeme!r})")

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF
