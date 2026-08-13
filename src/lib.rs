pub mod adc;
pub mod benchmark;
pub mod cli;
pub mod config;
pub mod error;
pub mod exact;
pub mod kmeans;
pub mod metrics;
pub mod pq;
pub mod report;
pub mod rng;
pub mod synthetic;
pub mod vector;

pub use error::{AppError, Result};
