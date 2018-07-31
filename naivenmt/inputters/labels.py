import collections


class Labels(collections.namedtuple(
  "labels",
  ("target_input_ids", "target_output_ids", "target_sequence_length"))):
  """Labels that feeds to Estimators's model_fn"""
  pass
