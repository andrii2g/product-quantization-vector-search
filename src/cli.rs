use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::{Result, synthetic::DatasetKind};

#[derive(Debug, Parser)]
#[command(name = "pqvs", about = "Product Quantization vector-search laboratory")]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run a preset benchmark matrix.
    Benchmark(BenchmarkArgs),
    /// Run one Product Quantization configuration.
    Run(RunArgs),
    /// Explain one query from an existing results file.
    InspectQuery(InspectQueryArgs),
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    #[arg(long, default_value = "default")]
    preset: String,
    #[arg(long, value_enum, default_value_t = DatasetKind::Clustered)]
    dataset: DatasetKind,
    #[arg(long, default_value = "artifacts")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(long, default_value_t = 64)]
    dimension: usize,
    #[arg(long, default_value_t = 10_000)]
    training_vectors: usize,
    #[arg(long, default_value_t = 50_000)]
    database_vectors: usize,
    #[arg(long, default_value_t = 200)]
    query_vectors: usize,
    #[arg(long, value_enum, default_value_t = DatasetKind::Clustered)]
    dataset: DatasetKind,
    #[arg(long, default_value_t = 8)]
    subspaces: usize,
    #[arg(long, default_value_t = 256)]
    centroids: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "artifacts")]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct InspectQueryArgs {
    #[arg(long, default_value = "artifacts/results.json")]
    experiment: PathBuf,
    #[arg(long, default_value_t = 0)]
    query: usize,
    #[arg(long)]
    subspaces: Option<usize>,
    #[arg(long)]
    centroids: Option<usize>,
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Benchmark(arguments) => crate::benchmark::run_benchmark_cli(
            &arguments.preset,
            arguments.dataset,
            &arguments.output,
        ),
        Command::Run(arguments) => {
            crate::benchmark::run_single_cli(crate::benchmark::SingleCliConfig {
                dimension: arguments.dimension,
                training_vectors: arguments.training_vectors,
                database_vectors: arguments.database_vectors,
                query_vectors: arguments.query_vectors,
                dataset: arguments.dataset,
                subspaces: arguments.subspaces,
                centroids: arguments.centroids,
                top_k: arguments.top_k,
                seed: arguments.seed,
                output: arguments.output,
            })
        }
        Command::InspectQuery(arguments) => crate::report::inspect_query(
            &arguments.experiment,
            arguments.query,
            arguments.subspaces,
            arguments.centroids,
        ),
    }
}
