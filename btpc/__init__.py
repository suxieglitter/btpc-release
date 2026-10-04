"""BTPC: Barlow Twins self-supervised P-wave polarity classification.

A two-stage, label-free pipeline for seismic P-wave polarity:

* Stage 1 trains a Barlow Twins encoder on two augmented views of each
  P-wave snippet and periodically filters training samples with an
  unsupervised reliability score.
* Stage 2 clusters encoder features (spectral clustering by default) into
  two pseudo-classes and trains a classification head on them.
* Validation and prediction map the A/B pseudo-classes to physical up/down
  polarities with a small labelled anchor set and reject unstable
  predictions.

Command line entry points: ``btpc-train``, ``btpc-valid``, ``btpc-predict``.
"""

__version__ = "0.1.0"
