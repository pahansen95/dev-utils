
import enum

from .protocols.intern import *

class Role(enum.StrEnum):
  PLATFORM = 'platform'
  DEVELOPER = 'developer'
  USER = 'user'
  AGENT = 'agent'
  OTHER = 'other'

