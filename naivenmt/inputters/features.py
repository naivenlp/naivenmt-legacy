import collections


class Features(collections.namedtuple(
  "features",
  ("source", "source_ids", "source_sequence_length"))):
  """Features that feeds to Estimator's model_fn."""
  pass


class ServingFeatures(collections.namedtuple(
  "serving_features",
  ("source", "source_ids", "source_sequence_length"))):
  """Features for serving input receiver"""
  pass
