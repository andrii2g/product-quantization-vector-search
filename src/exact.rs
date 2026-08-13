use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    vector::{Neighbor, TopK, VectorSet, squared_l2},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactSearchCounters {
    pub vectors_scanned: u64,
    pub coordinate_distance_operations: u64,
}

pub fn search_exact(
    database: &VectorSet,
    query: &[f32],
    top_k: usize,
    counters: &mut ExactSearchCounters,
) -> Result<Vec<Neighbor>> {
    if query.len() != database.dimension() {
        return Err(AppError::DimensionMismatch {
            expected: database.dimension(),
            actual: query.len(),
        });
    }
    if top_k == 0 || top_k > database.len() {
        return Err(AppError::InvalidConfig(
            "top-k must be positive and no larger than the database".into(),
        ));
    }
    let mut best = TopK::new(top_k);
    for vector_id in 0..database.len() {
        best.insert(Neighbor {
            vector_id,
            distance: squared_l2(database.vector(vector_id), query)?,
        });
        counters.vectors_scanned += 1;
        counters.coordinate_distance_operations += database.dimension() as u64;
    }
    Ok(best.into_sorted())
}

pub fn search_exact_batch(
    database: &VectorSet,
    queries: &VectorSet,
    top_k: usize,
) -> Result<(Vec<Vec<Neighbor>>, ExactSearchCounters)> {
    if queries.dimension() != database.dimension() {
        return Err(AppError::DimensionMismatch {
            expected: database.dimension(),
            actual: queries.dimension(),
        });
    }
    let mut counters = ExactSearchCounters::default();
    let mut results = Vec::with_capacity(queries.len());
    for query_id in 0..queries.len() {
        results.push(search_exact(
            database,
            queries.vector(query_id),
            top_k,
            &mut counters,
        )?);
    }
    Ok((results, counters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_search_matches_full_sort_and_counts_work() {
        let database = VectorSet::from_vectors(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ])
        .unwrap();
        let query = [1.2, 0.0];
        let mut reference: Vec<_> = (0..database.len())
            .map(|vector_id| Neighbor {
                vector_id,
                distance: squared_l2(database.vector(vector_id), &query).unwrap(),
            })
            .collect();
        reference.sort_by(Neighbor::compare_quality);
        reference.truncate(3);

        let mut counters = ExactSearchCounters::default();
        let actual = search_exact(&database, &query, 3, &mut counters).unwrap();
        assert_eq!(actual, reference);
        assert_eq!(counters.vectors_scanned, 4);
        assert_eq!(counters.coordinate_distance_operations, 8);
    }

    #[test]
    fn batch_counter_formula_is_exact() {
        let database = VectorSet::new(2, vec![0.0; 8]).unwrap();
        let queries = VectorSet::new(2, vec![0.0; 6]).unwrap();
        let (_, counters) = search_exact_batch(&database, &queries, 2).unwrap();
        assert_eq!(counters.vectors_scanned, 12);
        assert_eq!(counters.coordinate_distance_operations, 24);
    }

    #[test]
    fn ties_prefer_lower_vector_id() {
        let database = VectorSet::from_vectors(vec![vec![-1.0], vec![1.0]]).unwrap();
        let mut counters = ExactSearchCounters::default();
        let result = search_exact(&database, &[0.0], 1, &mut counters).unwrap();
        assert_eq!(result[0].vector_id, 0);
    }
}
