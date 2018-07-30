import collections


class Receivers(collections.namedtuple(
  "receivers",
  ("source", "source_sequence_length"))):
  """Receiver tensors for serving"""
  pass
