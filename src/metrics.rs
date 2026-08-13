use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    pq::{PqIndex, ProductQuantizer},
    vector::{Neighbor, VectorSet, squared_l2},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecallMetrics {
    pub recall_at_1: Option<f64>,
    pub recall_at_5: Option<f64>,
    pub recall_at_10: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub full_vector_bytes_per_vector: usize,
    pub actual_pq_code_bytes_per_vector: usize,
    pub theoretical_packed_bits_per_vector: usize,
    pub codebook_bytes: usize,
    pub amortized_codebook_bytes_per_vector: f64,
    pub amortized_pq_bytes_per_vector: f64,
    pub raw_compression_ratio: f64,
    pub amortized_compression_ratio: f64,
}

pub fn recall_at(exact: &[Neighbor], approximate: &[Neighbor], level: usize) -> Result<f64> {
    if level == 0 || exact.len() < level || approximate.len() < level {
        return Err(AppError::InvalidConfig(
            "recall level must be positive and available in both result lists".into(),
        ));
    }
    let hits = exact[..level]
        .iter()
        .filter(|exact_neighbor| {
            approximate[..level]
                .iter()
                .any(|candidate| candidate.vector_id == exact_neighbor.vector_id)
        })
        .count();
    Ok(hits as f64 / level as f64)
}

pub fn aggregate_recall(
    exact_queries: &[Vec<Neighbor>],
    approximate_queries: &[Vec<Neighbor>],
    top_k: usize,
) -> Result<RecallMetrics> {
    if exact_queries.is_empty() || exact_queries.len() != approximate_queries.len() {
        return Err(AppError::InvalidConfig(
            "exact and approximate query result counts must match and be non-empty".into(),
        ));
    }
    let average_at = |level: usize| -> Result<Option<f64>> {
        if top_k < level {
            return Ok(None);
        }
        let mut total = 0.0;
        for (exact, approximate) in exact_queries.iter().zip(approximate_queries) {
            total += recall_at(exact, approximate, level)?;
        }
        Ok(Some(total / exact_queries.len() as f64))
    };
    Ok(RecallMetrics {
        recall_at_1: average_at(1)?,
        recall_at_5: average_at(5)?,
        recall_at_10: average_at(10)?,
    })
}

pub fn mean_squared_reconstruction_error_per_vector(
    database: &VectorSet,
    quantizer: &ProductQuantizer,
    index: &PqIndex,
) -> Result<f64> {
    if database.len() != index.len() || database.dimension() != quantizer.dimension {
        return Err(AppError::InvalidConfig(
            "database, PQ index, and quantizer shapes must agree".into(),
        ));
    }
    let subvector_dimension = quantizer.subvector_dimension();
    let mut total = 0.0_f64;
    for vector_id in 0..database.len() {
        for subspace_index in 0..quantizer.subspaces {
            let start = subspace_index * subvector_dimension;
            let centroid_id = index.code(vector_id)[subspace_index] as usize;
            let centroid = quantizer.codebooks[subspace_index]
                .centroids
                .get(centroid_id)
                .ok_or_else(|| AppError::InvalidPqCode("centroid ID is out of range".into()))?;
            total += squared_l2(
                &database.vector(vector_id)[start..start + subvector_dimension],
                centroid,
            )? as f64;
        }
    }
    Ok(total / database.len() as f64)
}

pub fn memory_metrics(
    dimension: usize,
    subspaces: usize,
    centroids_per_subspace: usize,
    database_vector_count: usize,
) -> Result<MemoryMetrics> {
    if dimension == 0 || subspaces == 0 || centroids_per_subspace == 0 || database_vector_count == 0
    {
        return Err(AppError::InvalidConfig(
            "memory metric inputs must be positive".into(),
        ));
    }
    let bits_per_code =
        usize::BITS as usize - (centroids_per_subspace - 1).leading_zeros() as usize;
    let full_vector_bytes_per_vector = dimension * std::mem::size_of::<f32>();
    let actual_pq_code_bytes_per_vector = subspaces;
    let theoretical_packed_bits_per_vector = subspaces * bits_per_code;
    // M × K × (D / M) simplifies to K × D centroid coordinates.
    let codebook_bytes = centroids_per_subspace * dimension * std::mem::size_of::<f32>();
    let amortized_codebook_bytes_per_vector = codebook_bytes as f64 / database_vector_count as f64;
    let amortized_pq_bytes_per_vector =
        actual_pq_code_bytes_per_vector as f64 + amortized_codebook_bytes_per_vector;
    Ok(MemoryMetrics {
        full_vector_bytes_per_vector,
        actual_pq_code_bytes_per_vector,
        theoretical_packed_bits_per_vector,
        codebook_bytes,
        amortized_codebook_bytes_per_vector,
        amortized_pq_bytes_per_vector,
        raw_compression_ratio: full_vector_bytes_per_vector as f64
            / actual_pq_code_bytes_per_vector as f64,
        amortized_compression_ratio: full_vector_bytes_per_vector as f64
            / amortized_pq_bytes_per_vector,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pq::SubspaceCodebook;

    fn neighbors(ids: &[usize]) -> Vec<Neighbor> {
        ids.iter()
            .enumerate()
            .map(|(rank, vector_id)| Neighbor {
                vector_id: *vector_id,
                distance: rank as f32,
            })
            .collect()
    }

    #[test]
    fn recall_handles_complete_partial_and_zero_overlap() {
        let exact = neighbors(&[1, 2, 3, 4, 5]);
        assert_eq!(recall_at(&exact, &exact, 5).unwrap(), 1.0);
        assert_eq!(
            recall_at(&exact, &neighbors(&[1, 8, 3, 9, 5]), 5).unwrap(),
            0.6
        );
        assert_eq!(
            recall_at(&exact, &neighbors(&[6, 7, 8, 9, 10]), 5).unwrap(),
            0.0
        );
    }

    #[test]
    fn memory_formulas_match_acceptance_scenario() {
        let metrics = memory_metrics(64, 8, 256, 50_000).unwrap();
        assert_eq!(metrics.full_vector_bytes_per_vector, 256);
        assert_eq!(metrics.actual_pq_code_bytes_per_vector, 8);
        assert_eq!(metrics.theoretical_packed_bits_per_vector, 64);
        assert_eq!(metrics.codebook_bytes, 65_536);
        assert_eq!(metrics.raw_compression_ratio, 32.0);
        let expected = 8.0 + 65_536.0 / 50_000.0;
        assert!((metrics.amortized_pq_bytes_per_vector - expected).abs() < 1.0e-12);
    }

    #[test]
    fn reconstruction_error_uses_mean_total_squared_error_per_vector() {
        let database = VectorSet::from_vectors(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let quantizer = ProductQuantizer {
            dimension: 2,
            subspaces: 1,
            centroids_per_subspace: 1,
            codebooks: vec![SubspaceCodebook {
                subspace_index: 0,
                subvector_dimension: 2,
                centroids: vec![vec![1.0, 2.0]],
            }],
        };
        let index = PqIndex::new(1, 2, vec![0, 0]).unwrap();
        assert_eq!(
            mean_squared_reconstruction_error_per_vector(&database, &quantizer, &index).unwrap(),
            4.0
        );
    }
}
