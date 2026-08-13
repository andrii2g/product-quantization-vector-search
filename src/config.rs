use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{AppError, Result, synthetic::DatasetKind};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub kind: DatasetKind,
    pub dimension: usize,
    pub training_vectors: usize,
    pub database_vectors: usize,
    pub query_vectors: usize,
    pub latent_clusters: usize,
    pub cluster_stddev: f32,
    pub seed: u64,
    pub top_k: usize,
}

impl DatasetConfig {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.dimension, 32 | 64) {
            return Err(AppError::InvalidConfig(
                "dimension must be 32 or 64 for the MVP".into(),
            ));
        }
        if self.training_vectors == 0 {
            return Err(AppError::InvalidConfig(
                "training vector count must be positive".into(),
            ));
        }
        if self.query_vectors == 0 {
            return Err(AppError::InvalidConfig(
                "query vector count must be positive".into(),
            ));
        }
        if self.top_k == 0 {
            return Err(AppError::InvalidConfig("top-k must be positive".into()));
        }
        if self.database_vectors < self.top_k {
            return Err(AppError::InvalidConfig(
                "database vector count must be at least top-k".into(),
            ));
        }
        if self.latent_clusters == 0 {
            return Err(AppError::InvalidConfig(
                "latent cluster count must be positive".into(),
            ));
        }
        if !self.cluster_stddev.is_finite() || self.cluster_stddev < 0.0 {
            return Err(AppError::InvalidConfig(
                "cluster standard deviation must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KMeansConfig {
    pub max_iterations: usize,
    pub convergence_epsilon: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_epsilon: 1.0e-4,
        }
    }
}

impl KMeansConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(AppError::InvalidConfig(
                "k-means maximum iterations must be positive".into(),
            ));
        }
        if !self.convergence_epsilon.is_finite() || self.convergence_epsilon < 0.0 {
            return Err(AppError::InvalidConfig(
                "k-means convergence epsilon must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PqConfig {
    pub dimension: usize,
    pub subspaces: usize,
    pub centroids_per_subspace: usize,
    pub kmeans: KMeansConfig,
}

impl PqConfig {
    pub fn validate(&self, training_vectors: usize) -> Result<()> {
        if !matches!(self.dimension, 32 | 64) {
            return Err(AppError::InvalidConfig(
                "dimension must be 32 or 64 for the MVP".into(),
            ));
        }
        if !matches!(self.subspaces, 4 | 8) {
            return Err(AppError::InvalidConfig(
                "subspaces must be 4 or 8 for the MVP".into(),
            ));
        }
        if !self.dimension.is_multiple_of(self.subspaces) {
            return Err(AppError::InvalidConfig(
                "dimension must be divisible by subspaces".into(),
            ));
        }
        if !matches!(self.centroids_per_subspace, 16 | 64 | 256) {
            return Err(AppError::InvalidConfig(
                "centroids per subspace must be 16, 64, or 256".into(),
            ));
        }
        if self.centroids_per_subspace > u8::MAX as usize + 1 {
            return Err(AppError::InvalidConfig(
                "centroids per subspace cannot exceed 256".into(),
            ));
        }
        if training_vectors < self.centroids_per_subspace {
            return Err(AppError::InvalidConfig(
                "training vector count must be at least the centroid count".into(),
            ));
        }
        self.kmeans.validate()
    }

    pub fn subvector_dimension(&self) -> usize {
        self.dimension / self.subspaces
    }
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub dataset: DatasetConfig,
    pub pq: PqConfig,
    pub output: PathBuf,
}

impl RunConfig {
    pub fn validate(&self) -> Result<()> {
        self.dataset.validate()?;
        if self.dataset.dimension != self.pq.dimension {
            return Err(AppError::InvalidConfig(
                "dataset and PQ dimensions must match".into(),
            ));
        }
        self.pq.validate(self.dataset.training_vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsupported_dimension() {
        let config = DatasetConfig {
            kind: DatasetKind::Clustered,
            dimension: 12,
            training_vectors: 100,
            database_vectors: 100,
            query_vectors: 1,
            latent_clusters: 4,
            cluster_stddev: 0.2,
            seed: 42,
            top_k: 10,
        };
        assert!(config.validate().is_err());
    }
}
