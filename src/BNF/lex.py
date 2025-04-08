"""

Tokenize an EBNF Grammar using Regular Expressions.

"""
from __future__ import annotations
from .core import *

__all__ = [
  'TERMINAL_LOOKUP', 'Lexer',
]

# Structural Terminals

@dataclass
class EmptyLine(Terminal):
  regex: str = r'^\s+$'
  """A Line with only whitespace"""

@dataclass
class Whitespace(Terminal):
  regex: str = r'[ \t\r\f\b]+'
  """Match whitespace (excluding newlines)"""

@dataclass
class Newline(Terminal):
  regex: str = r'\n'
  """Explicit newline character"""

@dataclass
class Comment(Terminal):
  regex: str = r'(?:#|//)[^\n]*'
  """Match single-line comments starting with # or // to end of line"""

@dataclass
class MultilineComment(Terminal):
  regex: str = r'/\*[\s\S]*?\*/'
  """Match multi-line block comments between /* and */"""

# Special Tokens; Reserved Symbols & Words

@dataclass
class TerminalStart(Terminal):
  regex: str = r'<\s*TERMINAL\s*>'
  """Starts the Rule Section"""

@dataclass
class TerminalStop(Terminal):
  regex: str = r'<\s*/TERMINAL\s*>'
  """Stops the Rule Section"""

@dataclass
class RuleStart(Terminal):
  regex: str = r'<\s*RULE\s*>'
  """Starts the Rule Section"""

@dataclass
class RuleStop(Terminal):
  regex: str = r'<\s*/RULE\s*>'
  """Stops the Rule Section"""

@dataclass
class Equals(Terminal):
  regex: str = re.escape("=")
  """Match the equals sign in rule definitions"""

@dataclass
class Comma(Terminal):
  regex: str = re.escape(",")
  """Match the comma in sequence ranges"""

@dataclass
class Colon(Terminal):
  regex: str = re.escape(":")
  """Match the colon in inline rule definitions"""

@dataclass
class Pipe(Terminal):
  regex: str = re.escape("|")
  """Match the alternation operator |"""

@dataclass
class OpenParen(Terminal):
  regex: str = re.escape("(")
  """Match open parenthesis ("""

@dataclass
class CloseParen(Terminal):
  regex: str = re.escape(")")
  """Match close parenthesis )"""

@dataclass
class OpenSBracket(Terminal):
  regex: str = re.escape("[")
  """Match open square bracket ["""

@dataclass
class CloseSBracket(Terminal):
  regex: str = re.escape("]")
  """Match close square bracket ]"""

@dataclass
class OpenBrace(Terminal):
  regex: str = re.escape("{")
  """Match open brace {"""

@dataclass
class CloseBrace(Terminal):
  regex: str = re.escape("}")
  """Match close brace }"""

@dataclass
class Star(Terminal):
  regex: str = re.escape("*")
  """Match repetition operator * (zero or more)"""

@dataclass
class Plus(Terminal):
  regex: str = re.escape("+")
  """Match repetition operator + (one or more)"""

@dataclass
class QMark(Terminal):
  regex: str = re.escape("?")
  """Match optional operator ? (zero or one)"""

@dataclass
class Minus(Terminal):
  regex: str = re.escape("-")
  """Match exception operator -"""

@dataclass
class Ellipsis(Terminal):
  regex: str = re.escape("...")
  """Match ellipsis for external definitions"""

@dataclass
class NotImplemented(Terminal):
  regex: str = re.escape("TODO")
  """A placeholder for incomplete grammars"""

# Semantics

@dataclass
class Integer(Terminal):
  regex: str = r'[0-9]+'
  """A whole number"""

@dataclass
class Regex(Terminal):
  regex: str = r'/(?!<\s*/)(?:[^/\\]|\\.)+/[a-zA-Z]*'
  """Match regular expression patterns like /pattern/flags"""

@dataclass
class MultilineString(Terminal):
  regex: str = r'([\'"]{3})((?:[^\\]|\\[\s\S]|(?!\1).)*?)\1'
  """A sequence of characters (including unescaped newlines) quoted by matching quote characters"""

@dataclass
class String(Terminal):
  regex: str = r'([\'"])(?:\\.|(?!\1)[^\n])*?\1'
  """A sequence of characters (excluding unescaped newlines) quoted by matching quote characters"""

@dataclass
class Literal(Terminal):
  regex: str = r'[^\s,:=|(){}\[\]*.+?-]+'
  """Any Literal Value not including reserved symbols"""

@dataclass
class EscapedLiteral(Terminal):
  regex: str = r'`[^`]+`'
  """Reserved symbol treated as literal - either wrapped in backticks or prefixed with backslash"""

TERMINALS: list[Terminal] = [
  # Structural tokens first
  EmptyLine(),
  Whitespace(),
  Newline(),
  TerminalStart(),
  TerminalStop(),
  RuleStart(),
  RuleStop(),
  Comment(),
  MultilineComment(),

  # Reserved Symbols & Words
  Comma(),
  Equals(),
  Colon(),
  Pipe(),
  OpenParen(),
  CloseParen(),
  OpenSBracket(),
  CloseSBracket(),
  OpenBrace(),
  CloseBrace(),
  Star(),
  Plus(),
  QMark(),
  Minus(),
  Ellipsis(),
  NotImplemented(),

  # Semantic Tokens
  Integer(),
  Regex(),
  MultilineString(),
  String(),
  EscapedLiteral(),
  Literal(),
]
"""The list of terminals to use"""
TERMINAL_LOOKUP = { t.name: t for t in TERMINALS }

@dataclass
class Lexer:
  terminals: list[Terminal] = field(default_factory=lambda: TERMINALS)

  def _find_newlines(self, s: str) -> list[int]:
    i = 0
    found = []
    while i < len(s):
      if s[i] == '\\' and i + 1 < len(s) and s[i + 1] == 'n':
        # This is an escaped newline, skip it
        i += 2
      elif s[i] == '\n':
        # This is an actual newline
        found.append(i)
        i += 1
      else:
        i += 1
    return found

  def _sort_matches(self, matches: list[tuple[Terminal, Match]]) -> list[tuple[Terminal, Match]]:
    """Sorts a list of terminals by their type & """
    return list(sorted(
      matches,
      key=lambda item: (
        item[0].priority, # Terminal Priority
        len(item[1].value), # Size of match
      ),
      reverse=True, # Descending Order
    ))

  def _match(self, grammar: str, pos: int) -> tuple[Terminal, Match] | None:
    """Match the best terminal starting from the current position."""

    # Gather Matches
    matches: list[tuple[Terminal, Match]] = []
    for term in self.terminals:
      logger.debug(f'Scanning for Lexeme {term.name} @ pos {pos}')
      match = term.match(grammar, start=pos)
      if match is None: continue
      matches.append((term, match))

    if len(matches) <= 0: return None    
    # Sort Matches
    sorted_matches = self._sort_matches(matches)

    # Return the highest priority Match
    logger.debug(f'Found Lexemes @ {pos}/{len(grammar)}...\n{'\n'.join(
      f'{t.name}={m.value}'
      for t, m in matches
    )}')
    return sorted_matches.pop(0) # Pop so we drop the reference to the list

  def _tokenize(self, grammar: str) -> list[Token]:
    """Tokenizes the EBNF Written Grammar"""

    # Initialize things
    pos = 0
    lineno, colno = 1, 1
    tokens: list[Token] = []

    # Find the longest match
    while pos < len(grammar):
      # Get the best matching terminal & add it as a token
      match = self._match(
        grammar, pos
      )
      if match is None: raise LexError(f'No matching Terminal at Ln {lineno}, Col {colno}')
      terminal, (value, start, stop) = match
      assert start == pos
      assert stop > start
      tokens.append(Token(
        type(terminal),
        value, start, stop,
        lineno, colno
      ))

      # Update the Document Position
      pos = stop
      line_idx = self._find_newlines(value)
      line_count = len(line_idx)
      if line_count > 0:
        lineno += line_count
        ridx = line_idx[-1]
        colno = len(value) - ridx
      else:
        colno += len(value)
      
      logger.info(f'Matched `{terminal.name}` Token @ Ln {lineno}, Col {colno}...\n---\n{value}\n---')

    return tokens

  # def _transform(self, tokens: list[Token]) -> list[Token]:
  #   """Replace leading Whitespace as Ident Tokens"""
  #   ### Scan the token list for the following pattern: NEWLINE WHITESPACE !{ EMPTY_LINE, WHITESPACE, NEWLINE, SECT_START, SECT_STOP, COMMENT, MULTI_COMMENT }
  #   _tokens = []
  #   idx = 0
  #   while idx < len(tokens):
  #     if (
  #       (idx + 2) < len(tokens) # Stop evaluation before out of bounds
  #         and
  #       tokens[idx].kind == Newline
  #         and
  #       tokens[idx + 1].kind == Whitespace
  #         and
  #        tokens[idx + 2].kind not in { EmptyLine, Whitespace, Newline, SectStart, SectStop, Comment, MultiComment }
  #     ):
  #       _tokens.extend([
  #         tokens[idx],
  #         Token(Indent, tokens[idx+1].value, tokens[idx+1].start, tokens[idx+1].stop, tokens[idx+1].lineno, tokens[idx+1].colno),
  #         tokens[idx+2],
  #       ])
  #       idx += 3
  #     else:
  #       _tokens.append(tokens[idx])
  #       idx += 1

  #   return _tokens
  
  # def __call__(self, grammar: str) -> list[Token]:
  #   return self._transform(self._scan(
  #     '\n' + grammar.strip() + '\n' # Make sure the Document starts & ends w/ a newline
  #   ))

  def __call__(self, grammar: str) -> list[Token]:
    # Compile the rules
    for terminal in self.terminals: terminal.compile()

    # Tokenize
    return self._tokenize(grammar)
