use crate::{AppError, Result};

pub fn squared_l2(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(AppError::DimensionMismatch {
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut distance = 0.0_f32;
    for coordinate in 0..left.len() {
        let difference = left[coordinate] - right[coordinate];
        distance += difference * difference;
    }
    debug_assert!(distance.is_finite());
    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_known_squared_distance() {
        assert_eq!(squared_l2(&[0.0, 0.0], &[3.0, 4.0]).unwrap(), 25.0);
        assert_eq!(squared_l2(&[1.0, 2.0], &[1.0, 2.0]).unwrap(), 0.0);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        assert!(squared_l2(&[1.0], &[1.0, 2.0]).is_err());
    }
}
