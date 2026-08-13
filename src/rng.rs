pub const DISTRIBUTION_TAG: u64 = 0x4449_5354_5249_4255;
pub const TRAINING_TAG: u64 = 0x0054_5241_494E;
pub const DATABASE_TAG: u64 = 0x4441_5441_4241_5345;
pub const QUERY_TAG: u64 = 0x0051_5545_5249_4553;
pub const KMEANS_TAG: u64 = 0x4B4D_4541_4E53;

pub fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub fn derive_seed(master_seed: u64, tag: u64) -> u64 {
    mix64(master_seed ^ tag)
}

pub fn derive_kmeans_seed(
    master_seed: u64,
    dimension: usize,
    subspaces: usize,
    centroids_per_subspace: usize,
    subspace_index: usize,
) -> u64 {
    let mut seed = derive_seed(master_seed, KMEANS_TAG);
    for value in [
        dimension as u64,
        subspaces as u64,
        centroids_per_subspace as u64,
        subspace_index as u64,
    ] {
        seed = mix64(seed ^ value);
    }
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivation_is_stable_and_namespaced() {
        assert_eq!(derive_seed(42, TRAINING_TAG), derive_seed(42, TRAINING_TAG));
        assert_ne!(derive_seed(42, TRAINING_TAG), derive_seed(42, QUERY_TAG));
        assert_ne!(
            derive_kmeans_seed(42, 32, 4, 16, 0),
            derive_kmeans_seed(42, 32, 4, 16, 1)
        );
    }
}
