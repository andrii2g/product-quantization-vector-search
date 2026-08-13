use clap::ValueEnum;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    config::DatasetConfig,
    rng::{DATABASE_TAG, DISTRIBUTION_TAG, QUERY_TAG, TRAINING_TAG, derive_seed},
    vector::VectorSet,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum DatasetKind {
    #[default]
    Clustered,
    Correlated,
    Uniform,
}

impl std::fmt::Display for DatasetKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clustered => write!(formatter, "clustered"),
            Self::Correlated => write!(formatter, "correlated"),
            Self::Uniform => write!(formatter, "uniform"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DatasetBundle {
    pub training: VectorSet,
    pub database: VectorSet,
    pub queries: VectorSet,
}

pub fn generate(config: &DatasetConfig) -> Result<DatasetBundle> {
    config.validate()?;
    match config.kind {
        DatasetKind::Clustered => generate_clustered(config),
        DatasetKind::Correlated => generate_correlated(config),
        DatasetKind::Uniform => generate_uniform(config),
    }
}

fn normal(mean: f32, standard_deviation: f32) -> Normal<f32> {
    Normal::new(mean, standard_deviation).expect("validated normal distribution parameters")
}

fn generate_clustered(config: &DatasetConfig) -> Result<DatasetBundle> {
    let mut distribution_rng =
        ChaCha8Rng::seed_from_u64(derive_seed(config.seed, DISTRIBUTION_TAG));
    let standard_normal = normal(0.0, 1.0);
    let centers: Vec<f32> = (0..config.latent_clusters * config.dimension)
        .map(|_| standard_normal.sample(&mut distribution_rng))
        .collect();

    let make_set = |count: usize, tag: u64| -> Result<VectorSet> {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.seed, tag));
        let noise = normal(0.0, config.cluster_stddev);
        let mut data = Vec::with_capacity(count * config.dimension);
        for _ in 0..count {
            let cluster = rng.random_range(0..config.latent_clusters);
            let center = &centers[cluster * config.dimension..(cluster + 1) * config.dimension];
            data.extend(center.iter().map(|value| *value + noise.sample(&mut rng)));
        }
        VectorSet::new(config.dimension, data)
    };

    Ok(DatasetBundle {
        training: make_set(config.training_vectors, TRAINING_TAG)?,
        database: make_set(config.database_vectors, DATABASE_TAG)?,
        queries: make_set(config.query_vectors, QUERY_TAG)?,
    })
}

fn generate_uniform(config: &DatasetConfig) -> Result<DatasetBundle> {
    let make_set = |count: usize, tag: u64| -> Result<VectorSet> {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.seed, tag));
        let data = (0..count * config.dimension)
            .map(|_| rng.random_range(-1.0_f32..1.0_f32))
            .collect();
        VectorSet::new(config.dimension, data)
    };
    Ok(DatasetBundle {
        training: make_set(config.training_vectors, TRAINING_TAG)?,
        database: make_set(config.database_vectors, DATABASE_TAG)?,
        queries: make_set(config.query_vectors, QUERY_TAG)?,
    })
}

fn generate_correlated(config: &DatasetConfig) -> Result<DatasetBundle> {
    let latent_dimension = config.dimension / 4;
    let mut distribution_rng =
        ChaCha8Rng::seed_from_u64(derive_seed(config.seed, DISTRIBUTION_TAG));
    let standard_normal = normal(0.0, 1.0);
    let projection: Vec<f32> = (0..latent_dimension * config.dimension)
        .map(|_| standard_normal.sample(&mut distribution_rng) / (latent_dimension as f32).sqrt())
        .collect();

    let make_set = |count: usize, tag: u64| -> Result<VectorSet> {
        let mut rng = ChaCha8Rng::seed_from_u64(derive_seed(config.seed, tag));
        let noise = normal(0.0, 0.05);
        let mut data = Vec::with_capacity(count * config.dimension);
        let mut latent = vec![0.0; latent_dimension];
        for _ in 0..count {
            for value in &mut latent {
                *value = standard_normal.sample(&mut rng);
            }
            for coordinate in 0..config.dimension {
                let mut value = 0.0;
                for latent_coordinate in 0..latent_dimension {
                    value += latent[latent_coordinate]
                        * projection[latent_coordinate * config.dimension + coordinate];
                }
                data.push(value + noise.sample(&mut rng));
            }
        }
        VectorSet::new(config.dimension, data)
    };

    Ok(DatasetBundle {
        training: make_set(config.training_vectors, TRAINING_TAG)?,
        database: make_set(config.database_vectors, DATABASE_TAG)?,
        queries: make_set(config.query_vectors, QUERY_TAG)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(kind: DatasetKind) -> DatasetConfig {
        DatasetConfig {
            kind,
            dimension: 32,
            training_vectors: 32,
            database_vectors: 40,
            query_vectors: 4,
            latent_clusters: 4,
            cluster_stddev: 0.2,
            seed: 42,
            top_k: 10,
        }
    }

    #[test]
    fn all_generators_are_deterministic_and_finite() {
        for kind in [
            DatasetKind::Clustered,
            DatasetKind::Uniform,
            DatasetKind::Correlated,
        ] {
            let first = generate(&config(kind)).unwrap();
            let second = generate(&config(kind)).unwrap();
            assert_eq!(first, second);
            assert!(first.training.data().iter().all(|value| value.is_finite()));
            assert_eq!(first.database.len(), 40);
            assert_eq!(first.queries.dimension(), 32);
        }
    }

    #[test]
    fn query_count_is_isolated_from_other_streams() {
        let first = config(DatasetKind::Clustered);
        let mut second = first.clone();
        second.query_vectors = 8;
        let first_bundle = generate(&first).unwrap();
        let second_bundle = generate(&second).unwrap();
        assert_eq!(first_bundle.training, second_bundle.training);
        assert_eq!(first_bundle.database, second_bundle.database);
        assert_ne!(first_bundle.queries, second_bundle.queries);
    }

    #[test]
    fn different_seed_changes_data() {
        let first = config(DatasetKind::Uniform);
        let mut second = first.clone();
        second.seed += 1;
        assert_ne!(generate(&first).unwrap(), generate(&second).unwrap());
    }
}
