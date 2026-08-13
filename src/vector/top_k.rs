use super::Neighbor;

#[derive(Clone, Debug)]
pub struct TopK {
    capacity: usize,
    neighbors: Vec<Neighbor>,
}

impl TopK {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            neighbors: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, candidate: Neighbor) {
        debug_assert!(candidate.distance.is_finite());
        if self.capacity == 0 {
            return;
        }

        if let Some(existing) = self
            .neighbors
            .iter_mut()
            .find(|neighbor| neighbor.vector_id == candidate.vector_id)
        {
            if candidate.compare_quality(existing).is_lt() {
                *existing = candidate;
            }
            return;
        }

        if self.neighbors.len() < self.capacity {
            self.neighbors.push(candidate);
            return;
        }

        let worst_index = self
            .neighbors
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.compare_quality(right))
            .map(|(index, _)| index)
            .expect("a full TopK has a worst element");
        if candidate
            .compare_quality(&self.neighbors[worst_index])
            .is_lt()
        {
            self.neighbors[worst_index] = candidate;
        }
    }

    pub fn into_sorted(mut self) -> Vec<Neighbor> {
        self.neighbors
            .sort_by(|left, right| left.compare_quality(right));
        self.neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_best_candidates_with_deterministic_ties() {
        let mut top = TopK::new(3);
        for (vector_id, distance) in [(4, 1.0), (2, 1.0), (3, 0.5), (1, 1.0), (0, 3.0)] {
            top.insert(Neighbor {
                vector_id,
                distance,
            });
        }
        let ids: Vec<_> = top
            .into_sorted()
            .into_iter()
            .map(|neighbor| neighbor.vector_id)
            .collect();
        assert_eq!(ids, vec![3, 1, 2]);
    }

    #[test]
    fn does_not_duplicate_ids() {
        let mut top = TopK::new(2);
        top.insert(Neighbor {
            vector_id: 1,
            distance: 2.0,
        });
        top.insert(Neighbor {
            vector_id: 1,
            distance: 1.0,
        });
        assert_eq!(top.into_sorted().len(), 1);
    }
}
