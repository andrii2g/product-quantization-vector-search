use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    config::PqConfig,
    kmeans,
    rng::derive_kmeans_seed,
    vector::{VectorSet, squared_l2},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubspaceCodebook {
    pub subspace_index: usize,
    pub subvector_dimension: usize,
    pub centroids: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProductQuantizer {
    pub dimension: usize,
    pub subspaces: usize,
    pub centroids_per_subspace: usize,
    pub codebooks: Vec<SubspaceCodebook>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubspaceTrainingMetrics {
    pub subspace_index: usize,
    pub iterations: usize,
    pub final_inertia: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PqTrainingResult {
    pub quantizer: ProductQuantizer,
    pub subspace_metrics: Vec<SubspaceTrainingMetrics>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingCounters {
    pub vectors_encoded: u64,
    pub centroid_comparisons: u64,
    pub coordinate_distance_operations: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PqIndex {
    subspaces: usize,
    vector_count: usize,
    codes: Vec<u8>,
}

impl PqIndex {
    pub fn new(subspaces: usize, vector_count: usize, codes: Vec<u8>) -> Result<Self> {
        if subspaces == 0 || codes.len() != subspaces * vector_count {
            return Err(AppError::InvalidPqCode(
                "flat code length must equal vector_count × subspaces".into(),
            ));
        }
        Ok(Self {
            subspaces,
            vector_count,
            codes,
        })
    }

    pub fn subspaces(&self) -> usize {
        self.subspaces
    }

    pub fn len(&self) -> usize {
        self.vector_count
    }

    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }

    pub fn code(&self, vector_id: usize) -> &[u8] {
        let start = vector_id * self.subspaces;
        &self.codes[start..start + self.subspaces]
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
}

impl ProductQuantizer {
    pub fn subvector_dimension(&self) -> usize {
        self.dimension / self.subspaces
    }

    pub fn validate(&self) -> Result<()> {
        if self.subspaces == 0 || !self.dimension.is_multiple_of(self.subspaces) {
            return Err(AppError::InvalidConfig(
                "PQ dimension must be divisible by positive subspaces".into(),
            ));
        }
        if self.centroids_per_subspace == 0 || self.centroids_per_subspace > 256 {
            return Err(AppError::InvalidConfig(
                "PQ centroid count must be in 1..=256".into(),
            ));
        }
        if self.codebooks.len() != self.subspaces {
            return Err(AppError::InvalidConfig(
                "PQ must contain one codebook per subspace".into(),
            ));
        }
        for (subspace_index, codebook) in self.codebooks.iter().enumerate() {
            if codebook.subspace_index != subspace_index
                || codebook.subvector_dimension != self.subvector_dimension()
                || codebook.centroids.len() != self.centroids_per_subspace
                || codebook
                    .centroids
                    .iter()
                    .any(|centroid| centroid.len() != self.subvector_dimension())
            {
                return Err(AppError::InvalidConfig(
                    "PQ codebook layout is inconsistent".into(),
                ));
            }
        }
        Ok(())
    }
}

pub fn train(
    training: &VectorSet,
    config: &PqConfig,
    master_seed: u64,
) -> Result<PqTrainingResult> {
    config.validate(training.len())?;
    if training.dimension() != config.dimension {
        return Err(AppError::DimensionMismatch {
            expected: config.dimension,
            actual: training.dimension(),
        });
    }
    let subvector_dimension = config.subvector_dimension();
    let mut codebooks = Vec::with_capacity(config.subspaces);
    let mut subspace_metrics = Vec::with_capacity(config.subspaces);
    for subspace_index in 0..config.subspaces {
        let start = subspace_index * subvector_dimension;
        let mut subspace_data = Vec::with_capacity(training.len() * subvector_dimension);
        for vector_id in 0..training.len() {
            subspace_data
                .extend_from_slice(&training.vector(vector_id)[start..start + subvector_dimension]);
        }
        let points = VectorSet::new(subvector_dimension, subspace_data)?;
        let model = kmeans::train(
            &points,
            config.centroids_per_subspace,
            &config.kmeans,
            derive_kmeans_seed(
                master_seed,
                config.dimension,
                config.subspaces,
                config.centroids_per_subspace,
                subspace_index,
            ),
        )?;
        subspace_metrics.push(SubspaceTrainingMetrics {
            subspace_index,
            iterations: model.iterations,
            final_inertia: model.final_inertia,
        });
        codebooks.push(SubspaceCodebook {
            subspace_index,
            subvector_dimension,
            centroids: model.centroids,
        });
    }
    Ok(PqTrainingResult {
        quantizer: ProductQuantizer {
            dimension: config.dimension,
            subspaces: config.subspaces,
            centroids_per_subspace: config.centroids_per_subspace,
            codebooks,
        },
        subspace_metrics,
    })
}

pub fn encode(
    quantizer: &ProductQuantizer,
    database: &VectorSet,
) -> Result<(PqIndex, EncodingCounters)> {
    quantizer.validate()?;
    if database.dimension() != quantizer.dimension {
        return Err(AppError::DimensionMismatch {
            expected: quantizer.dimension,
            actual: database.dimension(),
        });
    }
    let subvector_dimension = quantizer.subvector_dimension();
    let mut codes = Vec::with_capacity(database.len() * quantizer.subspaces);
    let mut counters = EncodingCounters::default();
    for vector_id in 0..database.len() {
        for subspace_index in 0..quantizer.subspaces {
            let start = subspace_index * subvector_dimension;
            let subvector = &database.vector(vector_id)[start..start + subvector_dimension];
            let mut nearest_centroid = 0;
            let mut nearest_distance = f32::INFINITY;
            for (centroid_id, centroid) in quantizer.codebooks[subspace_index]
                .centroids
                .iter()
                .enumerate()
            {
                let distance = squared_l2(subvector, centroid)?;
                counters.centroid_comparisons += 1;
                counters.coordinate_distance_operations += subvector_dimension as u64;
                if distance < nearest_distance {
                    nearest_distance = distance;
                    nearest_centroid = centroid_id;
                }
            }
            codes.push(nearest_centroid as u8);
        }
        counters.vectors_encoded += 1;
    }
    Ok((
        PqIndex::new(quantizer.subspaces, database.len(), codes)?,
        counters,
    ))
}

pub fn reconstruct(quantizer: &ProductQuantizer, code: &[u8]) -> Result<Vec<f32>> {
    quantizer.validate()?;
    if code.len() != quantizer.subspaces {
        return Err(AppError::InvalidPqCode(
            "code must contain one centroid ID per subspace".into(),
        ));
    }
    let mut reconstructed = Vec::with_capacity(quantizer.dimension);
    for (subspace_index, centroid_id) in code.iter().enumerate() {
        let centroid = quantizer.codebooks[subspace_index]
            .centroids
            .get(*centroid_id as usize)
            .ok_or_else(|| AppError::InvalidPqCode("centroid ID is out of range".into()))?;
        reconstructed.extend_from_slice(centroid);
    }
    Ok(reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KMeansConfig;

    fn hand_quantizer() -> ProductQuantizer {
        ProductQuantizer {
            dimension: 4,
            subspaces: 2,
            centroids_per_subspace: 2,
            codebooks: vec![
                SubspaceCodebook {
                    subspace_index: 0,
                    subvector_dimension: 2,
                    centroids: vec![vec![0.0, 0.0], vec![2.0, 2.0]],
                },
                SubspaceCodebook {
                    subspace_index: 1,
                    subvector_dimension: 2,
                    centroids: vec![vec![10.0, 10.0], vec![20.0, 20.0]],
                },
            ],
        }
    }

    #[test]
    fn encoding_is_flat_deterministic_and_in_range() {
        let database =
            VectorSet::from_vectors(vec![vec![0.0, 0.0, 10.0, 10.0], vec![2.0, 2.0, 20.0, 20.0]])
                .unwrap();
        let quantizer = hand_quantizer();
        let (index, counters) = encode(&quantizer, &database).unwrap();
        assert_eq!(index.codes(), &[0, 0, 1, 1]);
        assert_eq!(index.codes().len(), database.len() * quantizer.subspaces);
        assert!(index.codes().iter().all(|id| *id < 2));
        assert_eq!(counters.vectors_encoded, 2);
    }

    #[test]
    fn reconstruction_concatenates_selected_centroids() {
        assert_eq!(
            reconstruct(&hand_quantizer(), &[1, 0]).unwrap(),
            vec![2.0, 2.0, 10.0, 10.0]
        );
    }

    #[test]
    fn trains_independent_deterministic_subspaces() {
        let mut vectors = Vec::new();
        for value in 0..16 {
            vectors.push(vec![value as f32; 32]);
        }
        let training = VectorSet::from_vectors(vectors).unwrap();
        let config = PqConfig {
            dimension: 32,
            subspaces: 4,
            centroids_per_subspace: 16,
            kmeans: KMeansConfig {
                max_iterations: 3,
                convergence_epsilon: 1.0e-4,
            },
        };
        let first = train(&training, &config, 42).unwrap();
        let second = train(&training, &config, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.quantizer.codebooks.len(), 4);
        assert_eq!(first.quantizer.codebooks[0].subvector_dimension, 8);
    }
}
