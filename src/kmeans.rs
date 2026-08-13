use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::{
    AppError, Result,
    config::KMeansConfig,
    vector::{VectorSet, squared_l2},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KMeansModel {
    pub centroids: Vec<Vec<f32>>,
    pub iterations: usize,
    pub final_inertia: f64,
}

pub fn initialize_kmeans_plus_plus(
    points: &VectorSet,
    centroid_count: usize,
    seed: u64,
) -> Result<Vec<Vec<f32>>> {
    validate_input(points, centroid_count)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let first_index = rng.random_range(0..points.len());
    let mut selected = vec![false; points.len()];
    selected[first_index] = true;
    let mut centroids = vec![points.vector(first_index).to_vec()];
    let mut nearest_distances = vec![f32::INFINITY; points.len()];

    while centroids.len() < centroid_count {
        let newest = centroids.last().expect("at least one centroid exists");
        for (point_index, nearest_distance) in nearest_distances.iter_mut().enumerate() {
            let distance = squared_l2(points.vector(point_index), newest)?;
            if distance < *nearest_distance {
                *nearest_distance = distance;
            }
        }
        let total_weight: f64 = nearest_distances.iter().map(|value| *value as f64).sum();
        let selected_index = if total_weight == 0.0 {
            selected
                .iter()
                .position(|is_selected| !is_selected)
                .ok_or_else(|| AppError::KMeans("no unselected point remains".into()))?
        } else {
            let draw = rng.random::<f64>() * total_weight;
            let mut cumulative = 0.0_f64;
            let mut chosen = points.len() - 1;
            for (point_index, weight) in nearest_distances.iter().enumerate() {
                cumulative += *weight as f64;
                if cumulative > draw {
                    chosen = point_index;
                    break;
                }
            }
            chosen
        };
        selected[selected_index] = true;
        centroids.push(points.vector(selected_index).to_vec());
    }
    Ok(centroids)
}

pub fn train(
    points: &VectorSet,
    centroid_count: usize,
    config: &KMeansConfig,
    seed: u64,
) -> Result<KMeansModel> {
    validate_input(points, centroid_count)?;
    config.validate()?;
    let mut centroids = initialize_kmeans_plus_plus(points, centroid_count, seed)?;
    let mut assignments = vec![0; points.len()];
    let mut assignment_distances = vec![0.0; points.len()];
    let epsilon_squared = config.convergence_epsilon * config.convergence_epsilon;
    let mut iterations = 0;

    for iteration in 0..config.max_iterations {
        let mut counts = assign_points(
            points,
            &centroids,
            &mut assignments,
            &mut assignment_distances,
        )?;
        repair_empty_clusters(
            points,
            &mut centroids,
            &mut assignments,
            &mut assignment_distances,
            &mut counts,
        )?;
        let new_centroids = recompute_centroids(points, centroid_count, &assignments, &counts);
        let mut maximum_shift = 0.0_f32;
        for centroid_index in 0..centroid_count {
            maximum_shift = maximum_shift.max(squared_l2(
                &centroids[centroid_index],
                &new_centroids[centroid_index],
            )?);
        }
        centroids = new_centroids;
        iterations = iteration + 1;
        if maximum_shift <= epsilon_squared {
            break;
        }
    }

    let final_inertia = points
        .data()
        .chunks_exact(points.dimension())
        .map(|point| {
            centroids
                .iter()
                .map(|centroid| squared_l2(point, centroid).expect("dimensions agree"))
                .min_by(f32::total_cmp)
                .expect("at least one centroid") as f64
        })
        .sum();
    Ok(KMeansModel {
        centroids,
        iterations,
        final_inertia,
    })
}

fn validate_input(points: &VectorSet, centroid_count: usize) -> Result<()> {
    if centroid_count == 0 {
        return Err(AppError::InvalidConfig(
            "k-means centroid count must be positive".into(),
        ));
    }
    if points.len() < centroid_count {
        return Err(AppError::InvalidConfig(
            "k-means needs at least as many points as centroids".into(),
        ));
    }
    Ok(())
}

fn assign_points(
    points: &VectorSet,
    centroids: &[Vec<f32>],
    assignments: &mut [usize],
    assignment_distances: &mut [f32],
) -> Result<Vec<usize>> {
    let mut counts = vec![0; centroids.len()];
    for point_index in 0..points.len() {
        let point = points.vector(point_index);
        let mut best_centroid = 0;
        let mut best_distance = squared_l2(point, &centroids[0])?;
        for (centroid_index, centroid) in centroids.iter().enumerate().skip(1) {
            let distance = squared_l2(point, centroid)?;
            if distance < best_distance {
                best_centroid = centroid_index;
                best_distance = distance;
            }
        }
        assignments[point_index] = best_centroid;
        assignment_distances[point_index] = best_distance;
        counts[best_centroid] += 1;
    }
    Ok(counts)
}

fn repair_empty_clusters(
    points: &VectorSet,
    centroids: &mut [Vec<f32>],
    assignments: &mut [usize],
    assignment_distances: &mut [f32],
    counts: &mut [usize],
) -> Result<()> {
    let empty_clusters: Vec<_> = counts
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect();
    for empty_cluster in empty_clusters {
        let donor_point = (0..points.len())
            .filter(|point_index| counts[assignments[*point_index]] > 1)
            .max_by(|left, right| {
                assignment_distances[*left]
                    .total_cmp(&assignment_distances[*right])
                    .then_with(|| right.cmp(left))
            })
            .ok_or_else(|| {
                AppError::KMeans("cannot repair an empty cluster without emptying a donor".into())
            })?;
        let source_cluster = assignments[donor_point];
        counts[source_cluster] -= 1;
        counts[empty_cluster] += 1;
        assignments[donor_point] = empty_cluster;
        assignment_distances[donor_point] = 0.0;
        centroids[empty_cluster].copy_from_slice(points.vector(donor_point));
    }
    Ok(())
}

fn recompute_centroids(
    points: &VectorSet,
    centroid_count: usize,
    assignments: &[usize],
    counts: &[usize],
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0_f64; points.dimension()]; centroid_count];
    for (point_index, cluster) in assignments.iter().enumerate() {
        for (sum, value) in sums[*cluster].iter_mut().zip(points.vector(point_index)) {
            *sum += *value as f64;
        }
    }
    sums.into_iter()
        .enumerate()
        .map(|(cluster, sum)| {
            sum.into_iter()
                .map(|value| (value / counts[cluster] as f64) as f32)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_and_training_are_deterministic() {
        let points = VectorSet::from_vectors(vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
        ])
        .unwrap();
        let config = KMeansConfig::default();
        assert_eq!(
            initialize_kmeans_plus_plus(&points, 2, 7).unwrap(),
            initialize_kmeans_plus_plus(&points, 2, 7).unwrap()
        );
        let first = train(&points, 2, &config, 7).unwrap();
        let second = train(&points, 2, &config, 7).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.centroids.len(), 2);
        assert!(first.iterations <= config.max_iterations);
    }

    #[test]
    fn repairs_empty_clusters_using_farthest_lowest_index() {
        let points =
            VectorSet::from_vectors(vec![vec![0.0], vec![2.0], vec![10.0], vec![20.0]]).unwrap();
        let mut centroids = vec![vec![0.0], vec![0.0], vec![0.0]];
        let mut assignments = vec![0, 0, 1, 1];
        let mut distances = vec![4.0, 4.0, 1.0, 1.0];
        let mut counts = vec![2, 2, 0];
        repair_empty_clusters(
            &points,
            &mut centroids,
            &mut assignments,
            &mut distances,
            &mut counts,
        )
        .unwrap();
        assert_eq!(assignments[0], 2);
        assert_eq!(counts, vec![1, 2, 1]);
        assert_eq!(centroids[2], vec![0.0]);
    }

    #[test]
    fn degenerate_initialization_allows_duplicate_values() {
        let points = VectorSet::from_vectors(vec![vec![1.0], vec![1.0], vec![1.0]]).unwrap();
        let centroids = initialize_kmeans_plus_plus(&points, 3, 1).unwrap();
        assert_eq!(centroids, vec![vec![1.0], vec![1.0], vec![1.0]]);
    }
}
