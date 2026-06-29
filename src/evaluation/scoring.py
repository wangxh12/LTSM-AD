from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReconstructionMahalanobisScorer:
    """Blend reconstruction error with a train-fitted multivariate normality score."""

    mean: np.ndarray
    precision: np.ndarray
    reconstruction_median: float
    reconstruction_mad: float
    mahalanobis_median: float
    mahalanobis_mad: float
    reconstruction_weight: float
    ridge_multiplier: float

    @classmethod
    def fit(
        cls,
        train_values: np.ndarray,
        validation_values: np.ndarray,
        validation_reconstruction_scores: np.ndarray,
        reconstruction_weight: float,
        ridge_multiplier: float,
    ) -> "ReconstructionMahalanobisScorer":
        if train_values.ndim != 2 or validation_values.ndim != 2:
            raise ValueError("train_values and validation_values must be rank-2 arrays")
        if train_values.shape[1] != validation_values.shape[1]:
            raise ValueError("Training and validation feature counts must match")
        if not 0.0 <= reconstruction_weight <= 1.0:
            raise ValueError("reconstruction_weight must be in [0, 1]")
        if ridge_multiplier <= 0.0:
            raise ValueError("ridge_multiplier must be positive")
        if len(validation_reconstruction_scores) != len(validation_values):
            raise ValueError("Validation reconstruction scores must align with validation values")

        mean = train_values.mean(axis=0)
        covariance = np.cov(train_values, rowvar=False)
        feature_count = covariance.shape[0]
        average_variance = max(float(np.trace(covariance)) / feature_count, 1e-12)
        ridge = average_variance * ridge_multiplier
        precision = np.linalg.pinv(covariance + ridge * np.eye(feature_count))
        validation_mahalanobis = cls._mahalanobis(validation_values, mean, precision)
        reconstruction_median, reconstruction_mad = cls._median_mad(validation_reconstruction_scores)
        mahalanobis_median, mahalanobis_mad = cls._median_mad(validation_mahalanobis)
        return cls(
            mean=mean.astype(np.float64),
            precision=precision.astype(np.float64),
            reconstruction_median=reconstruction_median,
            reconstruction_mad=reconstruction_mad,
            mahalanobis_median=mahalanobis_median,
            mahalanobis_mad=mahalanobis_mad,
            reconstruction_weight=float(reconstruction_weight),
            ridge_multiplier=float(ridge_multiplier),
        )

    @staticmethod
    def _median_mad(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise ValueError("Cannot calibrate a score without finite validation values")
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad < 1e-6:
            raise ValueError("Validation score MAD is zero; score normalization is undefined")
        return median, mad

    @staticmethod
    def _mahalanobis(values: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        centered = values - mean
        return np.sqrt(np.einsum("ni,ij,nj->n", centered, precision, centered))

    def score(self, reconstruction_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
        if len(reconstruction_scores) != len(values):
            raise ValueError("Reconstruction scores and values must have matching lengths")
        normalized_reconstruction = (reconstruction_scores - self.reconstruction_median) / self.reconstruction_mad
        mahalanobis = self._mahalanobis(values, self.mean, self.precision)
        normalized_mahalanobis = (mahalanobis - self.mahalanobis_median) / self.mahalanobis_mad
        return self.reconstruction_weight * normalized_reconstruction + (
            1.0 - self.reconstruction_weight
        ) * normalized_mahalanobis

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "reconstruction_mahalanobis",
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "reconstruction_median": self.reconstruction_median,
            "reconstruction_mad": self.reconstruction_mad,
            "mahalanobis_median": self.mahalanobis_median,
            "mahalanobis_mad": self.mahalanobis_mad,
            "reconstruction_weight": self.reconstruction_weight,
            "ridge_multiplier": self.ridge_multiplier,
        }


__all__ = ["ReconstructionMahalanobisScorer"]
