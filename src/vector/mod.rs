mod distance;
mod top_k;

pub use distance::squared_l2;
pub use top_k::TopK;

use serde::{Deserialize, Serialize};

use crate::{AppError, Result};

pub type VectorId = usize;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Neighbor {
    pub vector_id: VectorId,
    pub distance: f32,
}

impl Neighbor {
    pub fn compare_quality(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.vector_id.cmp(&other.vector_id))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VectorSet {
    dimension: usize,
    data: Vec<f32>,
}

impl VectorSet {
    pub fn new(dimension: usize, data: Vec<f32>) -> Result<Self> {
        if dimension == 0 {
            return Err(AppError::InvalidConfig(
                "vector dimension must be positive".into(),
            ));
        }
        if !data.len().is_multiple_of(dimension) {
            return Err(AppError::DimensionMismatch {
                expected: dimension,
                actual: data.len() % dimension,
            });
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(AppError::InvalidConfig(
                "vector coordinates must be finite".into(),
            ));
        }
        Ok(Self { dimension, data })
    }

    pub fn from_vectors(vectors: Vec<Vec<f32>>) -> Result<Self> {
        let dimension = vectors.first().map_or(0, Vec::len);
        if dimension == 0 {
            return Err(AppError::InvalidConfig(
                "vector set must contain non-empty vectors".into(),
            ));
        }
        let mut data = Vec::with_capacity(vectors.len() * dimension);
        for vector in vectors {
            if vector.len() != dimension {
                return Err(AppError::DimensionMismatch {
                    expected: dimension,
                    actual: vector.len(),
                });
            }
            data.extend(vector);
        }
        Self::new(dimension, data)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dimension
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn vector(&self, index: usize) -> &[f32] {
        let start = index * self.dimension;
        &self.data[start..start + self.dimension]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}
