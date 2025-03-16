"""

Concrete Chat Implementation

"""
from __future__ import annotations
from .core import *
from collections import deque

class Role(enum.StrEnum):
  PLATFORM = 'platform'
  """The entity providing the platform"""
  DEVELOPER = 'developer'
  """A developer modifying the platform"""
  USER = 'user'
  """A human user interacting w/ the platform"""
  AGENT = 'agent'
  """A non-human user interacting w/ the platform"""
  OTHER = 'other'
  """An entity of unknown origin interacting w/ the platform"""

class MessageProps(TypedDict, total=False):
  created_at: int
  """Unix Nanosecond Timestamp when the message was originally created"""
  author: NotRequired[int]
  """The Author of the Message"""
  agent_requested_version: str
  """The requested agent version (usually of a LLM)"""
  agent_provider_version: str
  """The actual agent version used by the provider (ie. the commit hash or snapshot date of an LLM)"""

class ChatMessage(TypedDict): # p.Message
  role: Role
  content: str
  props: NotRequired[MessageProps]

  @staticmethod
  def to_json(msg: ChatMessage) -> dict:
    spec = {
      'role': msg['role'].value,
      'content': msg['content'],
    }
    if 'props' in msg: spec['props'] = msg['props']
    return {
      'kind': 'ChatMessage',
      'spec': spec,
    }
  @staticmethod
  def marshal(o: ChatMessage) -> bytes: return json.dumps(ChatMessage.to_json(o)).encode()
  
  @staticmethod
  def from_json(obj: dict) -> ChatMessage:
    if not isinstance(obj, dict) or not {'kind', 'spec'}.issubset(obj.keys()): raise ValueError('Expected a JSON Object w/ Keys { .kind, .spec }')
    if (kind := obj['kind'] != 'ChatMessage'): raise ValueError(f'Expected Kind `ChatMessage` but got `{kind}`')
    spec = obj['spec']
    assert isinstance(spec, dict)
    msg = { k: v for k, v in spec.items() if k in { 'role', 'content', 'props' } }
    assert { 'role', 'content' }.issubset(msg.keys())
    msg['role'] = Role(msg['role'])
    return msg

  @staticmethod
  def unmarshal(buf: ByteString) -> ChatMessage: return ChatMessage.from_json(json.loads(buf))

def msg_fingerprint(msg: ChatMessage) -> str: return hashlib.md5(
  string=(msg['role'] + msg['content']).encode(),
  usedforsecurity=False,
).hexdigest()

ChatLog = list[ChatMessage]

@dataclass
class Conversation(p.Conversation, OrderedMultiTree[ChatMessage]):

  _: KW_ONLY
  _default_ekey: str = field(default='chat')

  @property
  def messages(self) -> set[ChatMessage]: return set(n.value for n in self.nodes.values())

  def __getitem__(self, mid: str) -> ChatMessage:
    if mid == self._root_loc: raise KeyError("getting the implicit root is not supported")
    return self._nodes[mid].value
  
  def new_chat(self, msg: ChatMessage) -> str:
    """Start a new chat log rooted at the passed message"""
    msg_loc = f"msg:0x{len(self._nodes):x}"
    msg_hash = msg_fingerprint(msg)
    self.insert_node(
      loc=msg_loc,
      val=msg, fngpnt=msg_hash,
      **{
        'created_at': msg['props']['created_at'],
      },
    )
    # Update Node Properties Seperately
    self._nodes[msg_loc].props |= msg['props']
    assert msg_loc in self._adjacent[self._default_ekey]

  def add_reply(self, msg: ChatMessage, to: str):
    """Add the message as a reply to the specified message identified by it's tree location"""
    msg_loc = f"msg:{len(self._nodes):x}"
    msg_hash = msg_fingerprint(msg)
    assert to in self._adjacent[self._default_ekey]
    if len(self._adjacent[self._default_ekey][to]) > 1: raise NotImplementedError('Branching Chats not supported')

    self.insert_node(
      loc=msg_loc, parent=to,
      val=msg, fngpnt=msg_hash,
      **{
        'created_at': msg['props']['created_at'],
      },
    )
    # Update Node Properties Seperately
    self._nodes[msg_loc].props |= msg['props']

  def chat_logs(self) -> Iterator[list[str]]:
    """A Lazy Iterator over all Chat Logs."""
    return iter( list(self.walk(root=log_root)) for log_root in self.children_of(self._root_loc) )
  
  @staticmethod
  def to_json(o: Conversation) -> dict:
    """Export the Conversation Tree a Json encodable map"""
    eks = list(o._edges.keys())
    def_ek = o._default_ekey
    root_loc = o._root_loc
    root_msgs = { ek: o.children_of(o._root_loc, key=ek) for ek in eks }

    node_defs: dict[str, dict] = {}
    for nid, node in o.nodes.items():
      assert nid not in node_defs
      node_defs[nid] = ChatMessage.to_json(node.value) | {
        'metadata': { 'id': nid, }
      }
      val_keys = frozenset(node.value['props'].keys())
      node_props = { k: v for k, v in node.props.items() if k not in val_keys }
      node_immutable = not node_props.pop('mutable')
      if node_immutable: node_props['mutable'] = False
      if node_props: node_defs[nid]['metadata']['props'] = node_props
    
    edge_defs: dict[str, list[dict]] = {}
    assert set(o._edges.keys()) == set(eks)
    for ek in eks:
      es = []
      for e in o.edges[ek]:
        es.append({
          'u': e.points[0],
          'v': e.points[1],
          'props': e.props,
        })
      edge_defs[ek] = es

    return {
      'kind': 'ConversationGraph',
      'metadata': {
        'props': {
          'default_edge_key': def_ek,
          'edge_keys': eks,
          'implicit_root': root_loc,
          'roots': root_msgs
        },
      },
      'spec': {
        'nodes': node_defs,
        'edges': edge_defs,
      }
    }

  @staticmethod
  def marshal(o: Conversation) -> bytes: return json.dumps(Conversation.to_json(o)).encode()

  @staticmethod
  def from_json(obj: dict) -> Conversation:
    if not isinstance(obj, dict): raise ValueError('Expected a JSON Object')
    if (kind := obj.get('kind', 'undefined')) != 'ConversationGraph':
      raise ValueError(f'Bad Kind: expected `ConversationGraph`: got `{kind}`')
    
    convo = Conversation(
      _root_loc=obj.get('metadata', {}).get('props', {}).get('implicit_root', ''),
      _default_ekey=obj.get('metadata', {}).get('props', {}).get('default_edge_key', 'chat'),
    )
    
    ### Process nodes
    node_defs: dict[str, dict] = obj['spec']['nodes']
    assert isinstance(node_defs, dict)
    for nid, node in node_defs.items():
      assert isinstance(node, dict)
      assert nid == node['metadata']['id']
      n_props = node['metadata'].get('props', {})
      if 'mutable' not in n_props: n_props['mutable'] = True
      n_val = ChatMessage.from_json(node)
      v_props = n_val.get('props', {})
      convo.add_node(loc=nid, val=n_val, fngpnt=nid, **(n_props | v_props))

    ### Process edges
    edge_defs: dict[str, list[dict]] = obj['spec']['edges']
    assert isinstance(edge_defs, dict)

    # Add Edge Keys
    for ek in set(obj.get('metadata', {}).get('props', {}).get('edge_keys', edge_defs.keys())):
      if ek != convo._default_ekey: convo.add_edge_key(ek)
    
    # Add Root Nodes
    for ek, nodes in obj.get('metadata', {}).get('props', {}).get('roots', {}).items():
      for nid in nodes:
        assert nid in convo._nodes.keys(), nid
        convo.add_edge(ek, (convo._root_loc, nid), True)

    # Add Remaining Nodes
    for ek, es in edge_defs.items():
      assert ek in convo._edges.keys(), ek
      for e in es:
        convo.add_edge(ek, (e['u'], e['v']), True, **e.get('props', {}))

    return convo

  @staticmethod
  def unmarshal(buf: ByteString) -> Conversation: return Conversation.from_json(json.loads(buf))
