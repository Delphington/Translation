# Транслятор арифметических выражений
# Даймидзенко Дмитрий, КМБО-05-23
#
# Грамматика (из задания 2, без левой рекурсии для LL(1)):
#
# <Start>   ::= <Expr>
# <Expr>    ::= <Term> <Expr'>
# <Expr'>   ::= + <Term> <Expr'> | - <Term> <Expr'> | ε
# <Term>    ::= <Factor> <Term'>
# <Term'>   ::= * <Factor> <Term'> | / <Factor> <Term'> | ε
# <Factor>  ::= <Factor2> <Factor'>
# <Factor'> ::= % <Factor2> <Factor'> | ε
# <Factor2> ::= ( <Expr> ) | N
#
# N - целое число (int)
# Терминалы: +, -, *, /, %, (, ), N

import sys
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Any, Optional


# типы токенов
class TokenType(Enum):
    PLUS = auto()  # +
    MINUS = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # %
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    NUM = auto()  # N - число
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, стр {self.line}, поз {self.col})"


class LexerError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"Лекс. ошибка [{line}:{col}]: {msg}")


# сканер - отдельный класс
class Lexer:

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1

    def curr(self):
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def advance(self):
        ch = self.curr()
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_spaces(self):
        while self.curr() and self.curr().isspace():
            self.advance()

    def read_num(self):
        ln, cl = self.line, self.col
        s = ''
        while self.curr() and self.curr().isdigit():
            s += self.advance()
        return Token(TokenType.NUM, int(s), ln, cl)

    def tokenize(self):
        tokens = []

        while self.curr() is not None:
            self.skip_spaces()
            if self.curr() is None:
                break

            ch = self.curr()
            ln, cl = self.line, self.col

            if ch.isdigit():
                tokens.append(self.read_num())
            elif ch == '+':
                self.advance()
                tokens.append(Token(TokenType.PLUS, '+', ln, cl))
            elif ch == '-':
                self.advance()
                tokens.append(Token(TokenType.MINUS, '-', ln, cl))
            elif ch == '*':
                self.advance()
                tokens.append(Token(TokenType.MUL, '*', ln, cl))
            elif ch == '/':
                self.advance()
                tokens.append(Token(TokenType.DIV, '/', ln, cl))
            elif ch == '%':
                self.advance()
                tokens.append(Token(TokenType.MOD, '%', ln, cl))
            elif ch == '(':
                self.advance()
                tokens.append(Token(TokenType.LPAREN, '(', ln, cl))
            elif ch == ')':
                self.advance()
                tokens.append(Token(TokenType.RPAREN, ')', ln, cl))
            else:
                raise LexerError(f"непонятный символ '{ch}'", ln, cl)

        tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return tokens


# узлы AST
class ASTNode:
    pass


@dataclass
class NumNode(ASTNode):
    value: int


@dataclass
class BinOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode


class ParserError(Exception):
    def __init__(self, msg, tok):
        super().__init__(f"Синт. ошибка [{tok.line}:{tok.col}]: {msg}")


# рекурсивный нисходящий парсер
# на каждый нетерминал своя функция
class Parser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def curr(self):
        return self.tokens[self.pos]

    def advance(self):
        t = self.curr()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def expect(self, tt):
        t = self.curr()
        if t.type != tt:
            raise ParserError(f"ожидался {tt.name}, получен {t.type.name}", t)
        return self.advance()

    def parse(self):
        return self.parse_start()

    # <Start> ::= <Expr>
    def parse_start(self):
        node = self.parse_expr()
        self.expect(TokenType.EOF)
        return node

    # <Expr> ::= <Term> <Expr'>
    def parse_expr(self):
        left = self.parse_term()
        return self.parse_expr_prime(left)

    # <Expr'> ::= + <Term> <Expr'> | - <Term> <Expr'> | ε
    def parse_expr_prime(self, left):
        t = self.curr()
        if t.type == TokenType.PLUS:
            self.advance()
            right = self.parse_term()
            node = BinOpNode(left, '+', right)
            return self.parse_expr_prime(node)
        elif t.type == TokenType.MINUS:
            self.advance()
            right = self.parse_term()
            node = BinOpNode(left, '-', right)
            return self.parse_expr_prime(node)
        else:
            # ε - ничего не делаем
            return left

    # <Term> ::= <Factor> <Term'>
    def parse_term(self):
        left = self.parse_factor()
        return self.parse_term_prime(left)

    # <Term'> ::= * <Factor> <Term'> | / <Factor> <Term'> | ε
    def parse_term_prime(self, left):
        t = self.curr()
        if t.type == TokenType.MUL:
            self.advance()
            right = self.parse_factor()
            node = BinOpNode(left, '*', right)
            return self.parse_term_prime(node)
        elif t.type == TokenType.DIV:
            self.advance()
            right = self.parse_factor()
            node = BinOpNode(left, '/', right)
            return self.parse_term_prime(node)
        else:
            return left

    # <Factor> ::= <Factor2> <Factor'>
    def parse_factor(self):
        left = self.parse_factor2()
        return self.parse_factor_prime(left)

    # <Factor'> ::= % <Factor2> <Factor'> | ε
    def parse_factor_prime(self, left):
        t = self.curr()
        if t.type == TokenType.MOD:
            self.advance()
            right = self.parse_factor2()
            node = BinOpNode(left, '%', right)
            return self.parse_factor_prime(node)
        else:
            return left

    # <Factor2> ::= ( <Expr> ) | N
    def parse_factor2(self):
        t = self.curr()
        if t.type == TokenType.LPAREN:
            self.advance()
            node = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return node
        elif t.type == TokenType.NUM:
            self.advance()
            return NumNode(t.value)
        else:
            raise ParserError(f"ожидалось число или '(', получен {t.type.name}", t)


class SemanticError(Exception):
    def __init__(self, msg):
        super().__init__(f"Сем. ошибка: {msg}")


# интерпретатор - вычисляет значение
class Interpreter:

    def eval(self, node):
        if isinstance(node, NumNode):
            return node.value

        elif isinstance(node, BinOpNode):
            left = self.eval(node.left)
            right = self.eval(node.right)

            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                if right == 0:
                    raise SemanticError("деление на ноль")
                return left // right  # целочисленное деление
            elif node.op == '%':
                if right == 0:
                    raise SemanticError("деление на ноль (остаток)")
                return left % right

        raise SemanticError(f"неизвестный узел {type(node).__name__}")


def print_ast(node, indent=0):
    pref = "  " * indent
    if isinstance(node, NumNode):
        print(f"{pref}Num: {node.value}")
    elif isinstance(node, BinOpNode):
        print(f"{pref}BinOp: {node.op}")
        print(f"{pref}  left:")
        print_ast(node.left, indent + 2)
        print(f"{pref}  right:")
        print_ast(node.right, indent + 2)


def main():
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                src = f.read()
        except FileNotFoundError:
            print(f"файл '{fname}' не найден")
            sys.exit(1)
        except IOError as e:
            print(f"ошибка чтения: {e}")
            sys.exit(1)
    else:
        print("Введите выражение (Ctrl+D для завершения):")
        try:
            src = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nотмена")
            sys.exit(0)

    if not src.strip():
        print("пустой ввод")
        sys.exit(1)

    try:
        # лексер
        print("\n--- Лексический анализ ---")
        lex = Lexer(src)
        tokens = lex.tokenize()
        print("Токены:")
        for t in tokens:
            print(f"  {t}")

        # парсер
        print("\n--- Синтаксический анализ ---")
        parser = Parser(tokens)
        ast = parser.parse()
        print("AST:")
        print_ast(ast)

        # вычисление
        print("\n--- Вычисление ---")
        interp = Interpreter()
        result = interp.eval(ast)
        print(f"Результат: {result}")

    except LexerError as e:
        print(f"\n{e}")
        sys.exit(1)

    except ParserError as e:
        print(f"\n{e}")
        sys.exit(1)

    except SemanticError as e:
        print(f"\n{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
