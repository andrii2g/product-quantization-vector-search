use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    pq::{PqIndex, ProductQuantizer},
    vector::{Neighbor, TopK, squared_l2},
};

#[derive(Clone, Debug, PartialEq)]
pub struct AdcDistanceTable {
    pub subspaces: usize,
    pub centroids_per_subspace: usize,
    pub distances: Vec<f32>,
}

impl AdcDistanceTable {
    pub fn distance(&self, subspace: usize, centroid_id: usize) -> f32 {
        self.distances[subspace * self.centroids_per_subspace + centroid_id]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdcTableCounters {
    pub centroid_distances: u64,
    pub coordinate_distance_operations: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdcSearchCounters {
    pub codes_scanned: u64,
    pub table_lookups: u64,
    pub distance_additions: u64,
}

pub fn build_distance_table(
    quantizer: &ProductQuantizer,
    query: &[f32],
    counters: &mut AdcTableCounters,
) -> Result<AdcDistanceTable> {
    quantizer.validate()?;
    if query.len() != quantizer.dimension {
        return Err(AppError::DimensionMismatch {
            expected: quantizer.dimension,
            actual: query.len(),
        });
    }
    let subvector_dimension = quantizer.subvector_dimension();
    let mut distances = Vec::with_capacity(quantizer.subspaces * quantizer.centroids_per_subspace);
    for subspace_index in 0..quantizer.subspaces {
        let start = subspace_index * subvector_dimension;
        let query_subvector = &query[start..start + subvector_dimension];
        for centroid in &quantizer.codebooks[subspace_index].centroids {
            distances.push(squared_l2(query_subvector, centroid)?);
            counters.centroid_distances += 1;
            counters.coordinate_distance_operations += subvector_dimension as u64;
        }
    }
    Ok(AdcDistanceTable {
        subspaces: quantizer.subspaces,
        centroids_per_subspace: quantizer.centroids_per_subspace,
        distances,
    })
}

pub fn code_distance(
    code: &[u8],
    table: &AdcDistanceTable,
    counters: &mut AdcSearchCounters,
) -> Result<f32> {
    if code.len() != table.subspaces {
        return Err(AppError::InvalidPqCode(
            "code and ADC table subspace counts differ".into(),
        ));
    }
    let mut distance = 0.0;
    for (subspace_index, centroid_id) in code.iter().enumerate() {
        if *centroid_id as usize >= table.centroids_per_subspace {
            return Err(AppError::InvalidPqCode(
                "centroid ID exceeds ADC table width".into(),
            ));
        }
        distance += table.distance(subspace_index, *centroid_id as usize);
        counters.table_lookups += 1;
        counters.distance_additions += 1;
    }
    Ok(distance)
}

pub fn search_adc(
    index: &PqIndex,
    table: &AdcDistanceTable,
    top_k: usize,
    counters: &mut AdcSearchCounters,
) -> Result<Vec<Neighbor>> {
    if index.subspaces() != table.subspaces {
        return Err(AppError::InvalidConfig(
            "PQ index and ADC table subspaces differ".into(),
        ));
    }
    if top_k == 0 || top_k > index.len() {
        return Err(AppError::InvalidConfig(
            "top-k must be positive and no larger than the PQ index".into(),
        ));
    }
    let mut best = TopK::new(top_k);
    for vector_id in 0..index.len() {
        let distance = code_distance(index.code(vector_id), table, counters)?;
        counters.codes_scanned += 1;
        best.insert(Neighbor {
            vector_id,
            distance,
        });
    }
    Ok(best.into_sorted())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pq::{SubspaceCodebook, reconstruct},
        vector::squared_l2,
    };

    fn quantizer() -> ProductQuantizer {
        ProductQuantizer {
            dimension: 2,
            subspaces: 2,
            centroids_per_subspace: 2,
            codebooks: vec![
                SubspaceCodebook {
                    subspace_index: 0,
                    subvector_dimension: 1,
                    centroids: vec![vec![0.0], vec![2.0]],
                },
                SubspaceCodebook {
                    subspace_index: 1,
                    subvector_dimension: 1,
                    centroids: vec![vec![10.0], vec![20.0]],
                },
            ],
        }
    }

    #[test]
    fn table_has_expected_layout_and_formula_counts() {
        let mut counters = AdcTableCounters::default();
        let table = build_distance_table(&quantizer(), &[1.0, 12.0], &mut counters).unwrap();
        assert_eq!(table.distances, vec![1.0, 1.0, 4.0, 64.0]);
        assert_eq!(counters.centroid_distances, 4);
        assert_eq!(counters.coordinate_distance_operations, 4);
    }

    #[test]
    fn scan_is_exhaustive_deterministic_and_counted() {
        let index = PqIndex::new(2, 4, vec![0, 0, 1, 0, 0, 1, 1, 1]).unwrap();
        let mut table_counters = AdcTableCounters::default();
        let table = build_distance_table(&quantizer(), &[1.0, 12.0], &mut table_counters).unwrap();
        let mut counters = AdcSearchCounters::default();
        let result = search_adc(&index, &table, 2, &mut counters).unwrap();
        assert_eq!(result[0].vector_id, 0);
        assert_eq!(result[1].vector_id, 1);
        assert_eq!(counters.codes_scanned, 4);
        assert_eq!(counters.table_lookups, 8);
        assert_eq!(counters.distance_additions, 8);
    }

    #[test]
    fn adc_equals_distance_to_reconstruction() {
        let quantizer = quantizer();
        let query = [1.0, 12.0];
        let mut table_counters = AdcTableCounters::default();
        let table = build_distance_table(&quantizer, &query, &mut table_counters).unwrap();
        for code in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            let mut scan_counters = AdcSearchCounters::default();
            let adc = code_distance(&code, &table, &mut scan_counters).unwrap();
            let reconstructed = reconstruct(&quantizer, &code).unwrap();
            let expected = squared_l2(&query, &reconstructed).unwrap();
            let tolerance = 1.0e-5 * expected.max(1.0);
            assert!((adc - expected).abs() <= tolerance);
        }
    }
}
