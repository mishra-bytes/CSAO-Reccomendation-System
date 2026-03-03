"""Feature pipeline orchestration.

This package re-exports the main feature pipeline from ``features.pipeline``.
The separation allows future addition of domain-specific sub-pipelines
(e.g. real-time feature computation, batch feature backfills).
"""

from features.pipeline import FeatureArtifacts, build_feature_artifacts, save_feature_artifacts

__all__ = ["FeatureArtifacts", "build_feature_artifacts", "save_feature_artifacts"]
