"""Test suite for the Sprout interpreter.

Uses only the standard-library `unittest` framework. Run from the repo root:

    python3 -m unittest discover -s tests

Layout mirrors the pipeline: lexer, parser, then end-to-end interpreter
behavior (the simplest way to assert on a tree-walker is to run a program
and inspect what it printed / what error it raised).
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout

# Make the project modules importable regardless of how the tests are invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast_nodes as ast
from errors import SproutError
from lexer import Lexer, LexError, TokenType
from parser import Parser, ParseError
from interpreter import Interpreter, RuntimeError_


# --- helpers --------------------------------------------------------------

def lex(source):
    """Return the list of Tokens for `source`."""
    return Lexer(source).tokenize()


def token_types(source):
    """Token types for `source`, excluding the trailing EOF."""
    return [t.type for t in lex(source)[:-1]]


def parse(source):
    """Return the list of top-level statements for `source`."""
    return Parser(lex(source)).parse()


def run_output(source):
    """Execute `source` and return everything it printed, as a list of lines."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        Interpreter().interpret(Parser(lex(source)).parse())
    text = buf.getvalue()
    return text.splitlines()


def run_io(source, stdin_text=""):
    """Like run_output, but feeds `stdin_text` to input(). Returns printed lines."""
    buf = io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin_text)
    try:
        with redirect_stdout(buf):
            Interpreter().interpret(Parser(lex(source)).parse())
    finally:
        sys.stdin = old_stdin
    return buf.getvalue().splitlines()


# --- lexer ----------------------------------------------------------------

class TestLexer(unittest.TestCase):
    def test_simple_assignment(self):
        self.assertEqual(
            token_types("let x = 5 + 3 * 2;"),
            [TokenType.LET, TokenType.IDENT, TokenType.ASSIGN, TokenType.NUMBER,
             TokenType.PLUS, TokenType.NUMBER, TokenType.STAR, TokenType.NUMBER,
             TokenType.SEMICOLON],
        )

    def test_two_char_operators(self):
        self.assertEqual(
            token_types("== != <= >= < > = !"),
            [TokenType.EQ, TokenType.NEQ, TokenType.LTE, TokenType.GTE,
             TokenType.LT, TokenType.GT, TokenType.ASSIGN, TokenType.BANG],
        )

    def test_line_comment_is_skipped(self):
        toks = lex("let y = 10; // a comment\nprint y;")
        self.assertNotIn(TokenType.SLASH, [t.type for t in toks])
        # `print` should be reported on line 2 after the newline.
        print_tok = next(t for t in toks if t.type == TokenType.PRINT)
        self.assertEqual(print_tok.line, 2)

    def test_slash_without_comment_is_division(self):
        self.assertEqual(token_types("6 / 2"),
                         [TokenType.NUMBER, TokenType.SLASH, TokenType.NUMBER])

    def test_number_literals_int_and_float(self):
        toks = lex("42 3.14")
        self.assertEqual(toks[0].literal, 42)
        self.assertIsInstance(toks[0].literal, int)
        self.assertEqual(toks[1].literal, 3.14)
        self.assertIsInstance(toks[1].literal, float)

    def test_string_literal_strips_quotes(self):
        self.assertEqual(lex('"hello"')[0].literal, "hello")

    def test_string_escape_sequences(self):
        # Source contains a backslash + letter; literal holds the real char.
        self.assertEqual(lex(r'"a\nb"')[0].literal, "a\nb")
        self.assertEqual(lex(r'"a\tb"')[0].literal, "a\tb")
        self.assertEqual(lex(r'"a\rb"')[0].literal, "a\rb")
        self.assertEqual(lex(r'"a\\b"')[0].literal, "a\\b")
        self.assertEqual(lex(r'"say \"hi\""')[0].literal, 'say "hi"')

    def test_escaped_quote_does_not_terminate(self):
        # Source is the 3 chars  " \ "  -> open quote, an escaped quote, then
        # EOF. The \" is consumed as a literal quote, so there is no closing
        # quote and the string is unterminated.
        with self.assertRaises(LexError):
            lex(r'"\"')

    def test_unknown_escape_raises(self):
        with self.assertRaises(LexError):
            lex(r'"bad\q"')

    def test_keywords_vs_identifiers(self):
        self.assertEqual(lex("let")[0].type, TokenType.LET)
        self.assertEqual(lex("lettuce")[0].type, TokenType.IDENT)

    def test_unexpected_character_raises(self):
        with self.assertRaises(LexError):
            lex("@")

    def test_unterminated_string_raises(self):
        with self.assertRaises(LexError):
            lex('"oops')


# --- parser ---------------------------------------------------------------

class TestParser(unittest.TestCase):
    def test_multiplication_binds_tighter_than_addition(self):
        stmt = parse("5 + 3 * 2;")[0]
        expr = stmt.expr
        self.assertIsInstance(expr, ast.Binary)
        self.assertEqual(expr.op, "+")
        self.assertIsInstance(expr.right, ast.Binary)
        self.assertEqual(expr.right.op, "*")

    def test_assignment_is_right_associative(self):
        # a = b = c  parses as  a = (b = c)
        expr = parse("a = b = c;")[0].expr
        self.assertIsInstance(expr, ast.Assign)
        self.assertEqual(expr.name, "a")
        self.assertIsInstance(expr.value, ast.Assign)
        self.assertEqual(expr.value.name, "b")

    def test_grouping_overrides_precedence(self):
        expr = parse("(5 + 3) * 2;")[0].expr
        self.assertEqual(expr.op, "*")
        self.assertIsInstance(expr.left, ast.Binary)
        self.assertEqual(expr.left.op, "+")

    def test_function_declaration_shape(self):
        stmt = parse("fn add(a, b) { return a + b; }")[0]
        self.assertIsInstance(stmt, ast.FunctionStmt)
        self.assertEqual(stmt.name, "add")
        self.assertEqual(stmt.params, ["a", "b"])

    def test_if_else_and_while(self):
        prog = parse("if (x) { print 1; } else { print 2; } while (y) { print 3; }")
        self.assertIsInstance(prog[0], ast.IfStmt)
        self.assertIsNotNone(prog[0].else_branch)
        self.assertIsInstance(prog[1], ast.WhileStmt)

    def test_chained_calls(self):
        expr = parse("f(1)(2);")[0].expr
        self.assertIsInstance(expr, ast.Call)
        self.assertIsInstance(expr.callee, ast.Call)

    def test_missing_semicolon_raises(self):
        with self.assertRaises(ParseError):
            parse("let x = 5")

    def test_invalid_assignment_target_raises(self):
        with self.assertRaises(ParseError):
            parse("5 = x;")

    def test_unexpected_eof_raises(self):
        with self.assertRaises(ParseError):
            parse("1 +;")


# --- interpreter (end to end) --------------------------------------------

class TestInterpreterExpressions(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        self.assertEqual(run_output("print 5 + 3 * 2;"), ["11"])

    def test_grouping(self):
        self.assertEqual(run_output("print (5 + 3) * 2;"), ["16"])

    def test_float_division(self):
        self.assertEqual(run_output("print 7 / 2;"), ["3.5"])

    def test_modulo(self):
        self.assertEqual(run_output("print 7 % 3;"), ["1"])

    def test_unary_minus_and_not(self):
        self.assertEqual(run_output("print -5; print !true; print !nil;"),
                         ["-5", "false", "true"])

    def test_comparison_and_equality(self):
        self.assertEqual(
            run_output("print 1 < 2; print 2 <= 2; print 3 == 3; print 3 != 4;"),
            ["true", "true", "true", "true"],
        )

    def test_string_concatenation(self):
        self.assertEqual(run_output('print "a" + "b";'), ["ab"])

    def test_nil_and_bool_stringify(self):
        self.assertEqual(run_output("print nil; print true; print false;"),
                         ["nil", "true", "false"])


class TestInterpreterControlFlow(unittest.TestCase):
    def test_truthiness_zero_is_truthy(self):
        self.assertEqual(run_output('if (0) { print "yes"; }'), ["yes"])

    def test_if_else_takes_else(self):
        self.assertEqual(run_output('if (false) { print "a"; } else { print "b"; }'),
                         ["b"])

    def test_while_loop(self):
        src = "let i = 0; while (i < 3) { print i; i = i + 1; }"
        self.assertEqual(run_output(src), ["0", "1", "2"])

    def test_logical_or_short_circuits(self):
        # boom() must never run, so its print never appears.
        src = ('fn boom() { print "BOOM"; return true; }'
               'if (true or boom()) { print "ok"; }')
        self.assertEqual(run_output(src), ["ok"])

    def test_logical_and_short_circuits(self):
        src = ('fn boom() { print "BOOM"; return true; }'
               'if (false and boom()) { print "no"; } print "done";')
        self.assertEqual(run_output(src), ["done"])

    def test_logical_returns_operand_value(self):
        self.assertEqual(run_output('print 1 and 2; print nil or "fallback";'),
                         ["2", "fallback"])


class TestInterpreterFunctions(unittest.TestCase):
    def test_function_call_and_return(self):
        self.assertEqual(run_output("fn sq(n) { return n * n; } print sq(9);"),
                         ["81"])

    def test_recursion(self):
        src = ("fn fib(n) { if (n < 2) { return n; } "
               "return fib(n - 1) + fib(n - 2); } print fib(10);")
        self.assertEqual(run_output(src), ["55"])

    def test_function_without_return_is_nil(self):
        self.assertEqual(run_output("fn noop() { } print noop();"), ["nil"])

    def test_closure_keeps_private_state(self):
        src = ("fn make() { let c = 0; fn inc() { c = c + 1; return c; } return inc; }"
               "let f = make(); print f(); print f(); print f();")
        self.assertEqual(run_output(src), ["1", "2", "3"])

    def test_block_scope_shadowing(self):
        src = ('let x = "outer"; { let x = "inner"; print x; } print x;')
        self.assertEqual(run_output(src), ["inner", "outer"])


class TestInterpreterForLoops(unittest.TestCase):
    def test_basic_counting(self):
        self.assertEqual(
            run_output("for (let i = 0; i < 4; i = i + 1) { print i; }"),
            ["0", "1", "2", "3"],
        )

    def test_iterate_array_by_index(self):
        src = ("let a = [10, 20, 30];"
               "for (let i = 0; i < len(a); i = i + 1) { print a[i]; }")
        self.assertEqual(run_output(src), ["10", "20", "30"])

    def test_loop_variable_does_not_leak(self):
        src = ("let i = 99; for (let i = 0; i < 2; i = i + 1) { } print i;")
        self.assertEqual(run_output(src), ["99"])

    def test_external_initializer(self):
        # No `let`: the for-loop reuses an existing variable.
        src = ("let i = 0; for (; i < 3; i = i + 1) { print i; } print i;")
        self.assertEqual(run_output(src), ["0", "1", "2", "3"])

    def test_omitted_increment(self):
        src = ("for (let i = 0; i < 3;) { print i; i = i + 1; }")
        self.assertEqual(run_output(src), ["0", "1", "2"])

    def test_return_breaks_out_of_loop(self):
        src = ("fn f() { for (let i = 0; i < 10; i = i + 1) { if (i == 2) { return i; } } return -1; }"
               "print f();")
        self.assertEqual(run_output(src), ["2"])

    def test_nested_for_loops(self):
        src = ("for (let i = 1; i <= 2; i = i + 1) {"
               "  for (let j = 1; j <= 2; j = j + 1) { print i * j; } }")
        self.assertEqual(run_output(src), ["1", "2", "2", "4"])


class TestInterpreterArrays(unittest.TestCase):
    def test_literal_and_print(self):
        self.assertEqual(run_output("print [1, 2, 3];"), ["[1, 2, 3]"])

    def test_indexing(self):
        self.assertEqual(run_output("let a = [10, 20, 30]; print a[1];"), ["20"])

    def test_negative_index(self):
        self.assertEqual(run_output("print [1, 2, 3][-1];"), ["3"])

    def test_index_assignment(self):
        self.assertEqual(run_output("let a = [1, 2, 3]; a[1] = 99; print a;"),
                         ["[1, 99, 3]"])

    def test_nested_arrays(self):
        self.assertEqual(run_output("print [[1, 2], [3, 4]][1][0];"), ["3"])

    def test_strings_quoted_inside_array(self):
        self.assertEqual(run_output('print ["x", "y"];'), ['["x", "y"]'])

    def test_empty_array_and_trailing_comma(self):
        self.assertEqual(run_output("print [];"), ["[]"])
        self.assertEqual(run_output("print [1, 2,];"), ["[1, 2]"])

    def test_len_builtin(self):
        self.assertEqual(run_output('print len([1, 2, 3]); print len("abcd");'),
                         ["3", "4"])

    def test_push_and_pop(self):
        src = "let a = [1]; push(a, 2); print a; print pop(a); print a;"
        self.assertEqual(run_output(src), ["[1, 2]", "2", "[1]"])

    def test_string_indexing(self):
        self.assertEqual(run_output('print "hello"[1];'), ["e"])

    def test_array_equality_by_value(self):
        self.assertEqual(run_output("print [1, 2] == [1, 2];"), ["true"])

    def test_index_out_of_range(self):
        with self.assertRaises(RuntimeError_):
            run_output("print [1, 2][5];")

    def test_non_integer_index(self):
        with self.assertRaises(RuntimeError_):
            run_output("print [1, 2][true];")

    def test_index_into_non_indexable(self):
        with self.assertRaises(RuntimeError_):
            run_output("print 5[0];")

    def test_pop_empty(self):
        with self.assertRaises(RuntimeError_):
            run_output("print pop([]);")


class TestInterpreterMaps(unittest.TestCase):
    def test_literal_and_lookup(self):
        src = 'let m = {"name": "sprout", "v": 1}; print m["name"]; print m["v"];'
        self.assertEqual(run_output(src), ["sprout", "1"])

    def test_update_existing_key(self):
        self.assertEqual(
            run_output('let m = {"a": 1}; m["a"] = 9; print m["a"];'), ["9"])

    def test_insert_new_key(self):
        self.assertEqual(
            run_output('let m = {"a": 1}; m["b"] = 2; print len(m); print m["b"];'),
            ["2", "2"])

    def test_number_keys(self):
        self.assertEqual(
            run_output('let m = {1: "one", 2: "two"}; print m[1]; print m[2];'),
            ["one", "two"])

    def test_bool_key(self):
        # Note: like Sprout's `==`, `true` and `1` are the same key (both
        # delegate to Python equality), so we test a bool key in isolation.
        self.assertEqual(run_output('let m = {false: "no"}; print m[false];'),
                         ["no"])

    def test_empty_map(self):
        self.assertEqual(run_output("let m = {}; print m; print len(m);"),
                         ["{}", "0"])

    def test_stringify_quotes_string_keys_and_values(self):
        self.assertEqual(run_output('print {"a": "b"};'), ['{"a": "b"}'])

    def test_nested_structures(self):
        self.assertEqual(
            run_output('print {"a": [1, 2]}["a"][1];'), ["2"])
        self.assertEqual(
            run_output('print [{"n": "x"}, {"n": "y"}][1]["n"];'), ["y"])

    def test_keys_builtin(self):
        self.assertEqual(run_output('print keys({"a": 1, "b": 2});'),
                         ['["a", "b"]'])

    def test_has_builtin(self):
        self.assertEqual(
            run_output('let m = {"a": 1}; print has(m, "a"); print has(m, "z");'),
            ["true", "false"])

    def test_trailing_comma(self):
        self.assertEqual(run_output('print len({"a": 1, "b": 2,});'), ["2"])

    def test_missing_key_raises(self):
        with self.assertRaises(RuntimeError_):
            run_output('let m = {"a": 1}; print m["z"];')

    def test_unhashable_key_in_literal_raises(self):
        with self.assertRaises(RuntimeError_):
            run_output("print {[1, 2]: 3};")

    def test_unhashable_key_in_assignment_raises(self):
        with self.assertRaises(RuntimeError_):
            run_output("let m = {}; m[[1]] = 2;")


class TestStandardLibrary(unittest.TestCase):
    def test_range_one_arg(self):
        self.assertEqual(run_output("print range(5);"), ["[0, 1, 2, 3, 4]"])

    def test_range_two_args(self):
        self.assertEqual(run_output("print range(2, 6);"), ["[2, 3, 4, 5]"])

    def test_range_with_step(self):
        self.assertEqual(run_output("print range(0, 10, 2);"), ["[0, 2, 4, 6, 8]"])

    def test_range_negative_step(self):
        self.assertEqual(run_output("print range(3, 0, -1);"), ["[3, 2, 1]"])

    def test_range_zero_step_raises(self):
        with self.assertRaises(RuntimeError_):
            run_output("print range(0, 5, 0);")

    def test_str_of_various(self):
        self.assertEqual(
            run_output('print str(42); print str(nil); print str([1, 2]);'),
            ["42", "nil", "[1, 2]"])

    def test_str_concatenation_use(self):
        # str() is what lets you build strings from numbers.
        self.assertEqual(run_output('print "n=" + str(5);'), ["n=5"])

    def test_num_int_and_float(self):
        self.assertEqual(run_output('print num("42"); print num("3.5");'),
                         ["42", "3.5"])

    def test_num_trims_whitespace_and_sign(self):
        self.assertEqual(run_output('print num("  -7  ");'), ["-7"])

    def test_num_of_number_is_identity(self):
        self.assertEqual(run_output("print num(9);"), ["9"])

    def test_num_invalid_raises(self):
        with self.assertRaises(RuntimeError_):
            run_output('print num("abc");')

    def test_split_on_separator(self):
        self.assertEqual(run_output('print split("a,b,c", ",");'),
                         ['["a", "b", "c"]'])

    def test_split_on_whitespace(self):
        self.assertEqual(run_output('print split("hi  there world");'),
                         ['["hi", "there", "world"]'])

    def test_join(self):
        self.assertEqual(run_output('print join(["a", "b", "c"], "-");'),
                         ["a-b-c"])

    def test_join_stringifies_elements(self):
        self.assertEqual(run_output('print join([1, 2, 3], ", ");'), ["1, 2, 3"])

    def test_split_join_round_trip(self):
        src = 'print join(split("a-b-c", "-"), "+");'
        self.assertEqual(run_output(src), ["a+b+c"])

    def test_input_reads_a_line(self):
        src = 'let n = input(); print "hi " + n;'
        self.assertEqual(run_io(src, "world\n"), ["hi world"])

    def test_input_then_num(self):
        src = 'let n = num(input()); print n + 1;'
        self.assertEqual(run_io(src, "41\n"), ["42"])

    def test_input_eof_returns_nil(self):
        src = 'let n = input(); print n;'
        self.assertEqual(run_io(src, ""), ["nil"])


class TestInterpreterErrors(unittest.TestCase):
    def test_division_by_zero(self):
        with self.assertRaises(RuntimeError_):
            run_output("print 1 / 0;")

    def test_modulo_by_zero(self):
        with self.assertRaises(RuntimeError_):
            run_output("print 1 % 0;")

    def test_undefined_variable(self):
        with self.assertRaises(RuntimeError_):
            run_output("print missing;")

    def test_type_mismatch_on_plus(self):
        with self.assertRaises(RuntimeError_):
            run_output('print 1 + "x";')

    def test_calling_non_function(self):
        with self.assertRaises(RuntimeError_):
            run_output("let x = 5; print x();")

    def test_wrong_arity(self):
        with self.assertRaises(RuntimeError_):
            run_output("fn f(a, b) { return a; } print f(1);")

    def test_bool_is_not_a_number(self):
        with self.assertRaises(RuntimeError_):
            run_output("print true * 2;")


class TestStringInterpolation(unittest.TestCase):
    def test_simple_variable(self):
        self.assertEqual(run_output('let n = "Sprout"; print "hi {n}";'),
                         ["hi Sprout"])

    def test_multiple_and_arithmetic(self):
        src = 'let a = 3; print "{a} + {a} = {a + a}";'
        self.assertEqual(run_output(src), ["3 + 3 = 6"])

    def test_number_is_stringified(self):
        self.assertEqual(run_output('let x = 42; print "x={x}";'), ["x=42"])

    def test_calls_and_indexing(self):
        src = ('fn sq(n) { return n * n; } let a = [10, 20];'
               'print "sq={sq(4)} first={a[0]} len={len(a)}";')
        self.assertEqual(run_output(src), ["sq=16 first=10 len=2"])

    def test_nested_string_inside_interpolation(self):
        src = 'let m = {"k": 7}; print "v={m["k"]}";'
        self.assertEqual(run_output(src), ["v=7"])

    def test_array_and_bool_values(self):
        self.assertEqual(run_output('print "{[1, 2]} {1 < 2}";'),
                         ["[1, 2] true"])

    def test_escaped_braces_are_literal(self):
        self.assertEqual(run_output(r'print "a \{b\} c";'), ["a {b} c"])

    def test_plain_string_unaffected(self):
        self.assertEqual(run_output('print "no braces here";'),
                         ["no braces here"])

    def test_leading_and_trailing_text(self):
        self.assertEqual(run_output('let x = 5; print "<{x}>";'), ["<5>"])

    def test_unterminated_interpolation_raises(self):
        with self.assertRaises(LexError):
            lex('"oops {name"')

    def test_empty_interpolation_raises(self):
        with self.assertRaises(LexError):
            lex('"{}"')

    def test_bad_expression_in_interpolation_raises(self):
        with self.assertRaises(ParseError):
            parse('print "{1 +}";')


class TestErrorMessages(unittest.TestCase):
    def test_format_points_caret_at_column(self):
        out = SproutError("bad", line=1, column=5).format("abcdefg", "Syntax error")
        src_line, caret_line = out.splitlines()[-2:]
        # The caret should sit directly under the 5th source character.
        self.assertIn("^", caret_line)
        self.assertEqual(caret_line.index("^"), src_line.index("abcdefg") + 4)

    def test_format_includes_line_number_and_message(self):
        out = SproutError("boom", line=3, column=2).format("a\nb\nccc", "Runtime error")
        self.assertIn("Runtime error: boom", out)
        self.assertIn("(line 3)", out)

    def test_format_without_line_is_plain(self):
        self.assertEqual(SproutError("oops").format("x", "Runtime error"),
                         "Runtime error: oops")

    def test_lex_error_carries_position(self):
        with self.assertRaises(LexError) as ctx:
            lex("let x = @;")
        self.assertEqual(ctx.exception.line, 1)
        self.assertEqual(ctx.exception.column, 9)  # '@' is the 9th column

    def test_parse_error_carries_position(self):
        with self.assertRaises(ParseError) as ctx:
            parse("let x = 5")  # missing ';'
        self.assertEqual(ctx.exception.line, 1)
        self.assertIsNotNone(ctx.exception.column)

    def test_runtime_error_carries_line(self):
        with self.assertRaises(RuntimeError_) as ctx:
            Interpreter().interpret(parse("print 1;\nprint 1 / 0;"))
        self.assertEqual(ctx.exception.line, 2)

    def test_runtime_error_line_inside_function(self):
        src = "fn f(n) {\n  return n / 0;\n}\nprint f(5);"
        with self.assertRaises(RuntimeError_) as ctx:
            Interpreter().interpret(parse(src))
        self.assertEqual(ctx.exception.line, 2)  # the division, not the call


if __name__ == "__main__":
    unittest.main()
