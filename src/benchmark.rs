use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    adc::{self, AdcSearchCounters, AdcTableCounters},
    config::{DatasetConfig, KMeansConfig, PqConfig},
    exact::search_exact_batch,
    metrics::{
        MemoryMetrics, RecallMetrics, aggregate_recall,
        mean_squared_reconstruction_error_per_vector, memory_metrics, recall_at,
    },
    pq::{self, EncodingCounters, SubspaceTrainingMetrics},
    synthetic::{DatasetBundle, DatasetKind, generate},
    vector::Neighbor,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Quick,
    Default,
}

impl Preset {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "quick" => Ok(Self::Quick),
            "default" => Ok(Self::Default),
            _ => Err(AppError::InvalidConfig(
                "preset must be 'quick' or 'default'".into(),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    pub dataset: DatasetConfig,
    pub pq_configs: Vec<PqConfig>,
    pub output: PathBuf,
}

impl BenchmarkConfig {
    pub fn from_preset(preset: Preset, kind: DatasetKind, output: PathBuf) -> Self {
        let (dimension, training, database, queries, clusters, centroid_counts) = match preset {
            Preset::Quick => (32, 1_000, 5_000, 25, 16, vec![16, 64]),
            Preset::Default => (64, 10_000, 50_000, 200, 32, vec![16, 64, 256]),
        };
        let mut pq_configs = Vec::new();
        for subspaces in [4, 8] {
            for centroids_per_subspace in &centroid_counts {
                pq_configs.push(PqConfig {
                    dimension,
                    subspaces,
                    centroids_per_subspace: *centroids_per_subspace,
                    kmeans: KMeansConfig::default(),
                });
            }
        }
        Self {
            dataset: DatasetConfig {
                kind,
                dimension,
                training_vectors: training,
                database_vectors: database,
                query_vectors: queries,
                latent_clusters: clusters,
                cluster_stddev: 0.2,
                seed: 42,
                top_k: 10,
            },
            pq_configs,
            output,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.dataset.validate()?;
        if self.pq_configs.is_empty() {
            return Err(AppError::InvalidConfig(
                "benchmark must contain at least one PQ configuration".into(),
            ));
        }
        for config in &self.pq_configs {
            if config.dimension != self.dataset.dimension {
                return Err(AppError::InvalidConfig(
                    "all PQ dimensions must match the dataset dimension".into(),
                ));
            }
            config.validate(self.dataset.training_vectors)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub generated_by: String,
    pub dataset: DatasetMetadata,
    pub exact: ExactBenchmarkResult,
    pub experiments: Vec<PqExperimentResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub kind: DatasetKind,
    pub dimension: usize,
    pub training_vectors: usize,
    pub database_vectors: usize,
    pub query_vectors: usize,
    pub latent_clusters: usize,
    pub seed: u64,
    pub top_k: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryNeighbors {
    pub query_id: usize,
    pub neighbors: Vec<Neighbor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExactBenchmarkResult {
    pub search_time_ms: f64,
    pub vectors_scanned: u64,
    pub coordinate_distance_operations: u64,
    pub queries: Vec<QueryNeighbors>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentConfigResult {
    pub subspaces: usize,
    pub centroids_per_subspace: usize,
    pub subvector_dimension: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub time_ms: f64,
    pub subspaces: Vec<SubspaceTrainingMetrics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingResult {
    pub time_ms: f64,
    pub vectors_encoded: u64,
    pub centroid_comparisons: u64,
    pub coordinate_distance_operations: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityResult {
    pub recall_at_1: Option<f64>,
    pub recall_at_5: Option<f64>,
    pub recall_at_10: Option<f64>,
    pub mean_squared_reconstruction_error_per_vector: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub adc_total_time_ms: f64,
    pub table_build_time_ms: f64,
    pub code_scan_time_ms: f64,
    pub table_centroid_distances: u64,
    pub table_coordinate_distance_operations: u64,
    pub codes_scanned: u64,
    pub table_lookups: u64,
    pub distance_additions: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApproximateQueryResult {
    pub query_id: usize,
    pub neighbors: Vec<Neighbor>,
    pub recall_at_k: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PqExperimentResult {
    pub config: ExperimentConfigResult,
    pub training: TrainingResult,
    pub encoding: EncodingResult,
    pub quality: QualityResult,
    pub memory: MemoryMetrics,
    pub search: SearchResult,
    pub queries: Vec<ApproximateQueryResult>,
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

pub fn run_pipeline(config: &BenchmarkConfig) -> Result<BenchmarkReport> {
    config.validate()?;
    let datasets = generate(&config.dataset)?;
    run_with_datasets(config, &datasets)
}

pub fn run_with_datasets(
    config: &BenchmarkConfig,
    datasets: &DatasetBundle,
) -> Result<BenchmarkReport> {
    config.validate()?;
    let exact_start = Instant::now();
    let (exact_neighbors, exact_counters) =
        search_exact_batch(&datasets.database, &datasets.queries, config.dataset.top_k)?;
    let exact_time_ms = milliseconds(exact_start);

    let mut experiments = Vec::with_capacity(config.pq_configs.len());
    for pq_config in &config.pq_configs {
        experiments.push(run_experiment(
            &config.dataset,
            pq_config,
            datasets,
            &exact_neighbors,
        )?);
    }
    Ok(BenchmarkReport {
        schema_version: 1,
        generated_by: format!("pqvs {}", env!("CARGO_PKG_VERSION")),
        dataset: DatasetMetadata {
            kind: config.dataset.kind,
            dimension: config.dataset.dimension,
            training_vectors: config.dataset.training_vectors,
            database_vectors: config.dataset.database_vectors,
            query_vectors: config.dataset.query_vectors,
            latent_clusters: config.dataset.latent_clusters,
            seed: config.dataset.seed,
            top_k: config.dataset.top_k,
        },
        exact: ExactBenchmarkResult {
            search_time_ms: exact_time_ms,
            vectors_scanned: exact_counters.vectors_scanned,
            coordinate_distance_operations: exact_counters.coordinate_distance_operations,
            queries: exact_neighbors
                .into_iter()
                .enumerate()
                .map(|(query_id, neighbors)| QueryNeighbors {
                    query_id,
                    neighbors,
                })
                .collect(),
        },
        experiments,
    })
}

pub fn run_experiment(
    dataset_config: &DatasetConfig,
    pq_config: &PqConfig,
    datasets: &DatasetBundle,
    exact_neighbors: &[Vec<Neighbor>],
) -> Result<PqExperimentResult> {
    let training_start = Instant::now();
    let training = pq::train(&datasets.training, pq_config, dataset_config.seed)?;
    let training_time_ms = milliseconds(training_start);

    let encoding_start = Instant::now();
    let (index, encoding_counters) = pq::encode(&training.quantizer, &datasets.database)?;
    let encoding_time_ms = milliseconds(encoding_start);

    let mut table_counters = AdcTableCounters::default();
    let mut scan_counters = AdcSearchCounters::default();
    let mut table_build_time_ms = 0.0;
    let mut code_scan_time_ms = 0.0;
    let mut approximate_neighbors = Vec::with_capacity(datasets.queries.len());
    for query_id in 0..datasets.queries.len() {
        let table_start = Instant::now();
        let table = adc::build_distance_table(
            &training.quantizer,
            datasets.queries.vector(query_id),
            &mut table_counters,
        )?;
        table_build_time_ms += milliseconds(table_start);
        let scan_start = Instant::now();
        approximate_neighbors.push(adc::search_adc(
            &index,
            &table,
            dataset_config.top_k,
            &mut scan_counters,
        )?);
        code_scan_time_ms += milliseconds(scan_start);
    }
    let recalls = aggregate_recall(
        exact_neighbors,
        &approximate_neighbors,
        dataset_config.top_k,
    )?;
    let reconstruction_error = mean_squared_reconstruction_error_per_vector(
        &datasets.database,
        &training.quantizer,
        &index,
    )?;
    let memory = memory_metrics(
        pq_config.dimension,
        pq_config.subspaces,
        pq_config.centroids_per_subspace,
        datasets.database.len(),
    )?;
    let query_results = approximate_neighbors
        .into_iter()
        .enumerate()
        .map(|(query_id, neighbors)| {
            Ok(ApproximateQueryResult {
                query_id,
                recall_at_k: recall_at(
                    &exact_neighbors[query_id],
                    &neighbors,
                    dataset_config.top_k,
                )?,
                neighbors,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(PqExperimentResult {
        config: ExperimentConfigResult {
            subspaces: pq_config.subspaces,
            centroids_per_subspace: pq_config.centroids_per_subspace,
            subvector_dimension: pq_config.subvector_dimension(),
        },
        training: TrainingResult {
            time_ms: training_time_ms,
            subspaces: training.subspace_metrics,
        },
        encoding: encoding_result(encoding_time_ms, encoding_counters),
        quality: quality_result(recalls, reconstruction_error),
        memory,
        search: SearchResult {
            adc_total_time_ms: table_build_time_ms + code_scan_time_ms,
            table_build_time_ms,
            code_scan_time_ms,
            table_centroid_distances: table_counters.centroid_distances,
            table_coordinate_distance_operations: table_counters.coordinate_distance_operations,
            codes_scanned: scan_counters.codes_scanned,
            table_lookups: scan_counters.table_lookups,
            distance_additions: scan_counters.distance_additions,
        },
        queries: query_results,
    })
}

fn encoding_result(time_ms: f64, counters: EncodingCounters) -> EncodingResult {
    EncodingResult {
        time_ms,
        vectors_encoded: counters.vectors_encoded,
        centroid_comparisons: counters.centroid_comparisons,
        coordinate_distance_operations: counters.coordinate_distance_operations,
    }
}

fn quality_result(recalls: RecallMetrics, reconstruction_error: f64) -> QualityResult {
    QualityResult {
        recall_at_1: recalls.recall_at_1,
        recall_at_5: recalls.recall_at_5,
        recall_at_10: recalls.recall_at_10,
        mean_squared_reconstruction_error_per_vector: reconstruction_error,
    }
}

pub struct SingleCliConfig {
    pub dimension: usize,
    pub training_vectors: usize,
    pub database_vectors: usize,
    pub query_vectors: usize,
    pub dataset: DatasetKind,
    pub subspaces: usize,
    pub centroids: usize,
    pub top_k: usize,
    pub seed: u64,
    pub output: PathBuf,
}

pub fn run_benchmark_cli(preset: &str, dataset: DatasetKind, output: &Path) -> Result<()> {
    let config =
        BenchmarkConfig::from_preset(Preset::parse(preset)?, dataset, output.to_path_buf());
    execute_and_write(&config)
}

pub fn run_single_cli(config: SingleCliConfig) -> Result<()> {
    let benchmark = BenchmarkConfig {
        dataset: DatasetConfig {
            kind: config.dataset,
            dimension: config.dimension,
            training_vectors: config.training_vectors,
            database_vectors: config.database_vectors,
            query_vectors: config.query_vectors,
            latent_clusters: 32,
            cluster_stddev: 0.2,
            seed: config.seed,
            top_k: config.top_k,
        },
        pq_configs: vec![PqConfig {
            dimension: config.dimension,
            subspaces: config.subspaces,
            centroids_per_subspace: config.centroids,
            kmeans: KMeansConfig::default(),
        }],
        output: config.output,
    };
    execute_and_write(&benchmark)
}

fn execute_and_write(config: &BenchmarkConfig) -> Result<()> {
    let report = run_pipeline(config)?;
    crate::report::write_artifacts(&config.output, &report)?;
    println!(
        "Exact search: {:.3} ms, {} coordinate operations",
        report.exact.search_time_ms, report.exact.coordinate_distance_operations
    );
    for experiment in &report.experiments {
        let recall = experiment
            .quality
            .recall_at_10
            .or(experiment.quality.recall_at_5)
            .or(experiment.quality.recall_at_1)
            .unwrap_or(0.0);
        println!(
            "PQ M={} K={}: Recall@{}={:.4}, raw compression={:.2}x, ADC time={:.3} ms",
            experiment.config.subspaces,
            experiment.config.centroids_per_subspace,
            report.dataset.top_k.min(10),
            recall,
            experiment.memory.raw_compression_ratio,
            experiment.search.adc_total_time_ms
        );
    }
    println!("Artifacts written to: {}", config.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_expand_in_stable_order() {
        let quick = BenchmarkConfig::from_preset(
            Preset::Quick,
            DatasetKind::Clustered,
            PathBuf::from("artifacts"),
        );
        let quick_pairs: Vec<_> = quick
            .pq_configs
            .iter()
            .map(|config| (config.subspaces, config.centroids_per_subspace))
            .collect();
        assert_eq!(quick_pairs, vec![(4, 16), (4, 64), (8, 16), (8, 64)]);

        let default = BenchmarkConfig::from_preset(
            Preset::Default,
            DatasetKind::Clustered,
            PathBuf::from("artifacts"),
        );
        let default_pairs: Vec<_> = default
            .pq_configs
            .iter()
            .map(|config| (config.subspaces, config.centroids_per_subspace))
            .collect();
        assert_eq!(
            default_pairs,
            vec![(4, 16), (4, 64), (4, 256), (8, 16), (8, 64), (8, 256)]
        );
    }

    #[test]
    fn tiny_experiment_reuses_supplied_exact_results() {
        let mut config = BenchmarkConfig::from_preset(
            Preset::Quick,
            DatasetKind::Uniform,
            PathBuf::from("unused"),
        );
        config.dataset.training_vectors = 16;
        config.dataset.database_vectors = 20;
        config.dataset.query_vectors = 2;
        config.dataset.latent_clusters = 4;
        config.pq_configs = vec![PqConfig {
            dimension: 32,
            subspaces: 4,
            centroids_per_subspace: 16,
            kmeans: KMeansConfig {
                max_iterations: 2,
                convergence_epsilon: 1.0e-4,
            },
        }];
        let report = run_pipeline(&config).unwrap();
        assert_eq!(report.experiments.len(), 1);
        assert_eq!(report.exact.vectors_scanned, 40);
        let experiment = &report.experiments[0];
        assert_eq!(experiment.search.codes_scanned, 40);
        assert_eq!(experiment.search.table_lookups, 160);
        assert_eq!(experiment.search.table_centroid_distances, 128);
        assert_eq!(
            experiment.search.table_coordinate_distance_operations,
            1_024
        );
    }
}
