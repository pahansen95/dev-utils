# Recursive Descent Parser

> Sourced from Wikipedia

In computer science, a **recursive descent parser** is a kind of top-down parser built from a set of mutually recursive procedures (or a non-recursive equivalent) where each such procedure implements one of the nonterminals of the grammar. Thus the structure of the resulting program closely mirrors that of the grammar it recognizes.

A *predictive parser* is a recursive descent parser that does not require backtracking. Predictive parsing is possible only for the class of LL(k) grammars, which are the context-free grammars for which there exists some positive integer *k* that allows a recursive descent parser to decide which production to use by examining only the next *k* tokens of input. The LL(k) grammars therefore exclude all ambiguous grammars, as well as all grammars that contain left recursion. Any context-free grammar can be transformed into an equivalent grammar that has no left recursion, but removal of left recursion does not always yield an LL(k) grammar. A predictive parser runs in linear time.

Recursive descent with backtracking is a technique that determines which production to use by trying each production in turn. Recursive descent with backtracking is not limited to LL(k) grammars, but is not guaranteed to terminate unless the grammar is LL(k). Even when they terminate, parsers that use recursive descent with backtracking may require exponential time.

Although predictive parsers are widely used, and are frequently chosen if writing a parser by hand, programmers often prefer to use a table-based parser produced by a parser generator, either for an LL(k) language or using an alternative parser, such as LALR or LR. This is particularly the case if a grammar is not in LL(k) form, as transforming the grammar to LL to make it suitable for predictive parsing is involved. Predictive parsers can also be automatically generated, using tools like ANTLR.

Predictive parsers can be depicted using transition diagrams for each non-terminal symbol where the edges between the initial and the final states are labelled by the symbols (terminals and non-terminals) of the right side of the production rule.

## Example Parser

The following EBNF-like grammar (for Niklaus Wirth's PL/0 programming language, from *Algorithms + Data Structures = Programs*) is in LL(1) form:

```ebnf
program = block “.” .

block =
[“const” ident “=” number {”,” ident “=” number} “;”]
[“var” ident {”,” ident} “;”]
{“procedure” ident “;” block “;”} statement .

statement =
ident “:=” expression
| “call” ident
| “begin” statement {”;” statement } “end”
| “if” condition “then” statement
| “while” condition “do” statement .

condition =
“odd” expression
| expression (”=”|”#”|”<”|”<=”|”>”|”>=”) expression .

expression = [”+”|”-”] term {(”+”|”-”) term} .

term = factor {(”*”|”/”) factor} .

factor =
ident
| number
| “(” expression “)” .
```

Terminals are expressed in quotes. Each nonterminal is defined by a rule in the grammar, except for *ident* and *number*, which are assumed to be implicitly defined.

### C Implementation

What follows is an implementation of a recursive descent parser for the above language in C. The parser reads in source code and exits with an error message if the code fails to parse, exiting silently if the code parses correctly.

Notice how closely the predictive parser below mirrors the grammar above. There is a procedure for each nonterminal in the grammar. Parsing descends in a top-down manner until the final nonterminal has been processed. The program fragment depends on a global variable, *sym*, which contains the current symbol from the input, and the function *nextsym*, which updates *sym* when called.

The implementations of the functions *nextsym* and *error* are omitted for simplicity.

```c
typedef enum {ident, number, lparen, rparen, times, slash, plus,
    minus, eql, neq, lss, leq, gtr, geq, callsym, beginsym, semicolon,
    endsym, ifsym, whilesym, becomes, thensym, dosym, constsym, comma,
    varsym, procsym, period, oddsym} Symbol;

Symbol sym;
void nextsym(void);
void error(const char msg[]);

int accept(Symbol s) {
    if (sym == s) {
        nextsym();
        return 1;
    }
    return 0;
}

int expect(Symbol s) {
    if (accept(s))
        return 1;
    error("expect: unexpected symbol");
    return 0;
}

void factor(void) {
    if (accept(ident)) {
        ;
    } else if (accept(number)) {
        ;
    } else if (accept(lparen)) {
        expression();
        expect(rparen);
    } else {
        error("factor: syntax error");
        nextsym();
    }
}

void term(void) {
    factor();
    while (sym == times || sym == slash) {
        nextsym();
        factor();
    }
}

void expression(void) {
    if (sym == plus || sym == minus)
        nextsym();
    term();
    while (sym == plus || sym == minus) {
        nextsym();
        term();
    }
}

void condition(void) {
    if (accept(oddsym)) {
        expression();
    } else {
        expression();
        if (sym == eql || sym == neq || sym == lss || sym == leq || sym == gtr || sym == geq) {
            nextsym();
            expression();
        } else {
            error("condition: invalid operator");
            nextsym();
        }
    }
}

void statement(void) {
    if (accept(ident)) {
        expect(becomes);
        expression();
    } else if (accept(callsym)) {
        expect(ident);
    } else if (accept(beginsym)) {
        do {
            statement();
        } while (accept(semicolon));
        expect(endsym);
    } else if (accept(ifsym)) {
        condition();
        expect(thensym);
        statement();
    } else if (accept(whilesym)) {
        condition();
        expect(dosym);
        statement();
    } else {
        error("statement: syntax error");
        nextsym();
    }
}

void block(void) {
    if (accept(constsym)) {
        do {
            expect(ident);
            expect(eql);
            expect(number);
        } while (accept(comma));
        expect(semicolon);
    }
    if (accept(varsym)) {
        do {
            expect(ident);
        } while (accept(comma));
        expect(semicolon);
    }
    while (accept(procsym)) {
        expect(ident);
        expect(semicolon);
        block();
        expect(semicolon);
    }
    statement();
}

void program(void) {
    nextsym();
    block();
    expect(period);
}
```