"""
Parse an EBNF+ Grammar into a Parse Tree.
"""
from __future__ import annotations
from .core import *

Rule_ = Rule
Terminal_ = Terminal

__all__ = [
  'RULE_LOOKUP', 'Parser'
]

@dataclass
class Parser:
  tokens: list[Token]
  
  def _filter_tokens(self, tokens: list[Token] = None) -> list[Token]:
    if tokens is None: tokens = self.tokens
    return [
      t for t in self.tokens
      if t.kind.name not in {
        'EMPTY_LINE', 'WHITESPACE', 'NEWLINE',
        'COMMENT', 'MULTILINE_COMMENT',
      }
    ]

  def __call__(self,
    entrypoint: type[Rule_] | Rule_ = None,
    tree: ParseTree = None,
    buffer: TokenBuffer = None,
  ) -> ParseTree:
    """Parses the tokenized EBNF into a Parse Tree"""
    buffer = None
    if entrypoint is None: entrypoint = Grammar
    if isinstance(entrypoint, type): # A Class Object
      if buffer is None: buffer = TokenBuffer(self._filter_tokens())
      if tree is None: tree = ParseTree()
    else: # A Class Instance Object
      assert isinstance(entrypoint, Rule_)
      if buffer is not None: raise ValueError("Can't pass a custom buffer w/ an initialized rule")
      if tree is not None: raise ValueError("Can't pass a custom tree w/ an initialized rule")
      buffer = entrypoint.buffer
      tree = entrypoint.tree
    assert tree is not None and buffer is not None
    # start_rule = self._initialize_rules(entrypoint, tree, buffer)
    start_rule = entrypoint(tree, buffer)
    start_rule(tree.root)
    return tree

@dataclass
class Grammar(Rule_):
  """grammar = terminal_section rule_section
  A sequence with one terminal section and one rule section
  """
  _: KW_ONLY
  terminal_section: type[TerminalSection] = field(default_factory=lambda: TerminalSection)
  rule_section: type[RuleSection] = field(default_factory=lambda: RuleSection)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Process sections based on their start tokens
    sections_processed: set[type[TerminalSection | RuleSection]] = set()
    
    while sections_processed != { RuleSection, TerminalSection }:
      if self.is_empty(): break
      
      token = self.peek_expected('TERMINAL_START', 'RULE_START')
      
      # Select the appropriate section rule
      section_rule = (
        self.terminal_section.from_rule(self)
          if token.kind.name == 'TERMINAL_START' else
        self.rule_section.from_rule(self)
      )

      # Check for duplicate sections
      if type(section_rule) in sections_processed:
        self.parse_error(f"Duplicate section: {token.kind.name}")
      else:
        yield section_rule
        sections_processed.add(type(section_rule))
    
    # Verify all required sections were processed
    if sections_processed != { RuleSection, TerminalSection }:
      self.parse_error("Missing sections - both TERMINAL and RULE sections are required")

@dataclass
class TerminalSection(Rule_):
  """terminal_section: TERMINAL_START terminal_definition+ TERMINAL_STOP
  A Section of terminal definitions
  """
  terminal_definition: type[TerminalDefinition] = field(default_factory=lambda: TerminalDefinition)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Check for terminal section start token
    yield self.expect_token('TERMINAL_START')
    
    # We need at least one definition
    self.raise_if_empty("Unexpected end of input after TERMINAL_START")
    
    # Process terminal definitions until we hit the section end
    total_defs = 0
    while not self.is_empty():
      
      # Check if we've reached the end of the section
      if self.next_matches('TERMINAL_STOP'): break
        
      # Process a terminal definition
      yield self.terminal_definition.from_rule(self)
      total_defs += 1
    
    # Ensure we saw at least one definition
    if total_defs <= 0: 
      self.parse_error("Terminal section must contain at least one definition")
    
    logger.debug(f'Processed {total_defs} rules in the Terminal Section')
    
    # Check for section end token
    yield self.expect_token('TERMINAL_STOP')

@dataclass
class RuleSection(Rule_):
  """rule_section: RULE_START rule_definition+ RULE_STOP
  A Section of rule definitions
  """
  rule_definition: type[RuleDefinition] = field(default_factory=lambda: RuleDefinition)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Check for rule section start token
    yield self.expect_token('RULE_START')
    
    # We need at least one definition
    self.raise_if_empty("Expected at least one rule definition after section start")
    
    total_defs = 0
    
    # Process rule definitions until we hit the section end
    while not self.is_empty():
      # Check if we've reached the end of the section
      if self.next_matches('RULE_STOP'): break
        
      # Process a terminal definition
      yield self.rule_definition.from_rule(self)
      total_defs += 1
    
    # Ensure we saw at least one definition
    if total_defs <= 0: 
      self.parse_error("Rule Section must contain at least one definition")
    
    logger.debug(f'Processed {total_defs} rules in the Rule Section')
    
    # Check for section end token
    yield self.expect_token('RULE_STOP')

# Definitions

@dataclass
class TerminalDefinition(Rule_):
  """terminal_definition: (LITERAL `:` MULTILINE_STRING? (STRING | REGEX | `...`)) |
  (LITERAL `=` (STRING | REGEX | `...`) MULTILINE_STRING?)
  """
  
  def _handle_inline_form(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the inline form: LITERAL = (STRING | REGEX | ...)"""
    yield self.consume()  # Consume EQUALS (already verified)
    
    # Check for STRING, REGEX, or ELLIPSIS
    yield self.expect_token('STRING', 'REGEX', 'ELLIPSIS')
    
    # Check for optional multiline string
    optional_doc = self.optional_token('MULTILINE_STRING')
    if optional_doc: yield optional_doc
  
  def _handle_expanded_form(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the expanded form: LITERAL : MULTILINE_STRING? (STRING | REGEX | ...)"""
    yield self.consume()  # Consume COLON (already verified)
    
    # Check for optional multiline string
    optional_doc = self.optional_token('MULTILINE_STRING')
    if optional_doc: yield optional_doc
    
    # Check for STRING, REGEX, or ELLIPSIS
    yield self.expect_token('STRING', 'REGEX', 'ELLIPSIS')
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Must start with a literal
    yield self.expect_token('LITERAL')
    
    # Check if we have a colon (expanded form) or equals (inline form)
    token = self.peek_expected('COLON', 'EQUALS')
    
    # Handle expanded form: LITERAL : MULTILINE_STRING? (STRING | REGEX | ...)
    if token.kind.name == 'COLON':
      yield from self._handle_expanded_form()
    
    # Handle inline form: LITERAL = (STRING | REGEX | ...) MULTILINE_STRING?
    else:
      yield from self._handle_inline_form()

@dataclass
class RuleDefinition(Rule_):
  """rule_definition: (LITERAL `:` MULTILINE_STRING? (statement | `...`)) |
  (LITERAL `=` (statement | `...`) MULTILINE_STRING?)
  """
  statement: type[Statement] = field(default_factory=lambda: Statement)
  
  def _handle_inline_form(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the inline form: LITERAL = (statement | ...) MULTILINE_STRING?"""
    yield self.consume()  # Consume EQUALS (already verified)
    
    # Check for statement or ELLIPSIS
    if self.next_matches('ELLIPSIS'): yield self.consume()
    else: yield self.statement.from_rule(self)
    
    # Check for optional multiline string
    optional_doc = self.optional_token('MULTILINE_STRING')
    if optional_doc: yield optional_doc
  
  def _handle_expanded_form(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the expanded form: LITERAL : MULTILINE_STRING? (statement | ...)"""
    yield self.consume()  # Consume COLON (already verified)
    
    # Check for optional multiline string
    optional_doc = self.optional_token('MULTILINE_STRING')
    if optional_doc:
      yield optional_doc
    
    # Check for statement or ELLIPSIS
    if self.next_matches('ELLIPSIS'):
      yield self.consume()
    else:
      yield self.statement.from_rule(self)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Must start with a literal
    yield self.expect_token('LITERAL')
    
    # Check if we have a colon (expanded form) or equals (inline form)
    token = self.peek_expected('COLON', 'EQUALS')
    
    # Handle expanded form: LITERAL : MULTILINE_STRING? (statement | ...)
    if token.kind.name == 'COLON':
      yield from self._handle_expanded_form()
    
    # Handle inline form: LITERAL = (statement | ...) MULTILINE_STRING?
    else:
      yield from self._handle_inline_form()

# Core Syntax

@dataclass
class Grouping(Rule_):
  """grouping = `(` statement `)`
  Establish hierarchical scoping between statements
  """
  statement: type[Statement] = field(default_factory=lambda: Statement)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Expect open parenthesis
    yield self.expect_token('OPEN_PAREN')
    
    # Process the statement
    yield self.statement.from_rule(self)
    
    # Expect close parenthesis
    yield self.expect_token('CLOSE_PAREN')

@dataclass
class Statement(Rule_):
  """A high-level syntactic construct that can handle either many clauses or alternatives'''

  statement:
    clause+ | # A Concatenation of Clauses
    clause (`|` clause)+ # An Alternation of Clauses
  """
  clause: type[Clause] = field(default_factory=lambda: Clause)
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token]:
    _clause_rule = self.clause.from_rule(self)
    
    # First, get the initial clause
    yield _clause_rule

    # Check for an Alternation
    if self.next_matches('PIPE'):
      while self.next_matches('PIPE'):
        yield self.consume()  # Consume PIPE
        yield _clause_rule
      return
    
    # If we reach here, then we need to attempt implicit concatanation
    while True:
      result = self.try_rule(
        type(_clause_rule),
        self.save_state(),
      )
      if result is None: break
      _, ptree, new_state = result
      self.load_state(new_state)
      yield ptree
    
@dataclass
class Clause(Rule_):
  """A collection of phrases...

  clause:
    phrase+ | # A Concatenation of Phrases
    ( phrase `-` phrase ) # A Phrase Exclusion
  """
  phrase: type[Phrase] = field(default_factory=lambda: Phrase)
  range: type[Range] = field(default_factory=lambda: Range)

  def __iter__(self) -> Iterator[ParseTree | Rule | Token]:
    _phrase_rule = self.phrase.from_rule(self)

    # Get the first phrase
    yield _phrase_rule
    
    # Check for exclusion
    if self.next_matches('MINUS'):
      yield self.consume()  # Consume MINUS
      yield _phrase_rule # Consume the 2nd Phrase
      return
    
    # otherwise try to parse another phrase for implicit concatenation
    while True:
      result = self.try_rule(
        type(_phrase_rule),
        self.save_state(),
      )
      if result is None: break
      _, ptree, new_state = result
      self.load_state(new_state)
      yield ptree

@dataclass
class Range(Rule_):
  """range = `{` INTEGER `}` | `{` INTEGER? , INTEGER? `}`
  Example: range = foobar{1} | foobar{,10} | foobar{1,} | foobar{1,10} | foobar{,}
  """
  
  def _handle_exact_range(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the {n} pattern - exact count"""
    yield self.consume()  # Consume INTEGER
    yield self.expect_token('CLOSE_BRACE')
  
  def _handle_min_max_range(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the {n,m} pattern - min to max range"""
    yield self.consume()  # Consume INTEGER
    yield self.expect_token('COMMA')
    yield self.expect_token('INTEGER')
    yield self.expect_token('CLOSE_BRACE')
  
  def _handle_min_range(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the {n,} pattern - min to infinity range"""
    yield self.consume()  # Consume INTEGER
    yield self.expect_token('COMMA')
    yield self.expect_token('CLOSE_BRACE')
  
  def _handle_max_range(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the {,m} pattern - 0 to max range"""
    yield self.expect_token('COMMA')
    yield self.expect_token('INTEGER')
    yield self.expect_token('CLOSE_BRACE')
  
  def _handle_any_range(self) -> Iterator[ParseTree | Rule | Token | None]:
    """Handle the {,} pattern - 0 to infinity range (any number)"""
    yield self.expect_token('COMMA')
    yield self.expect_token('CLOSE_BRACE')
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    # Expect opening brace
    yield self.expect_token('OPEN_BRACE')
    if self.match_pattern('INTEGER', 'LITERAL', 'INTEGER'): yield from self._handle_min_max_range()
    elif self.match_pattern('LITERAL', 'INTEGER'): yield from self._handle_max_range()
    elif self.match_pattern('INTEGER', 'LITERAL'): yield from self._handle_min_range()
    elif self.match_pattern('INTEGER'): yield from self._handle_exact_range()
    elif self.match_pattern('LITERAL'): yield from self._handle_any_range()
    else: self.parse_error('Bad Range')

@dataclass
class Phrase(Rule_):
  """phrase: (lexeme+ | grouping) ( range | `?` | `*` | `+` )?
  A basic sequence of terminals or some hierarchical unit
  """
  lexeme: type[Lexeme] = field(default_factory=lambda: Lexeme)
  grouping: type[Grouping] = field(default_factory=lambda: Grouping)
  range: type[Range] = field(default_factory=lambda: Range)

  def _optional_modifier(self) -> Iterator[Rule | Token | None]:
    self.log('Searching for Optional Modifier')
    if self.next_matches('{'): yield self.range.from_rule(self)
    elif self.next_matches('QMARK', 'STAR', 'PLUS'): yield self.consume()
    else: yield None
  
  def _valid_literal(self) -> bool: return (
    self.next_matches('LITERAL', 'ESCAPED_LITERAL')
      and
    not self.next_matches('EQUALS', 'COLON', lookahead=1) # Don't match Rule Names
  )
  
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:

    if self.next_matches('OPEN_PAREN'):
      yield self.grouping.from_rule(self)
      yield from self._optional_modifier()

    elif self._valid_literal():
      _lexeme_rule = self.lexeme.from_rule(self)
      while self._valid_literal(): yield _lexeme_rule
      yield from self._optional_modifier()

    else:
      self.parse_error('Expected a Group or a Sequence of Literals')
      assert False
    
@dataclass
class Lexeme(Rule_):
  """lexeme: LITERAL | ESCAPED_LITERAL
  Some value representing a terminal in the described grammar
  """
  def __iter__(self) -> Iterator[ParseTree | Rule | Token | None]:
    yield self.expect_token('LITERAL', 'ESCAPED_LITERAL')

RULES: list[Rule_] = [
  Grammar,
  TerminalSection,
  RuleSection,
  TerminalDefinition,
  RuleDefinition,
  Grouping,
  Statement,
  Clause,
  Range,
  Phrase,
  Lexeme
]
RULE_LOOKUP = { r.name: r for r in RULES }
