import collections


class Labels(collections.namedtuple(
  "labels",
  ("target", "target_ids", "target_sequence_length"))):
  """Labels that feeds to Estimators's model_fn"""
  pass
