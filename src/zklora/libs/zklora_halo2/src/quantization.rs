use halo2_proofs::{arithmetic::Field, pasta::Fp};
use num_traits::{Zero, FromPrimitive};
use std::ops::{Add, Mul, Sub};
use crate::field_traits::WrappedFp;
use ff::PrimeField;

/// Represents a fixed-point number in the field
#[derive(Clone, Debug)]
pub struct FixedPoint<F: Field + FromPrimitive + Zero + PrimeField> {
    /// The scaled integer value
    value: F,
    /// The scale factor (power of 2)
    scale_bits: u32,
}

impl<F: Field + FromPrimitive + Zero + PrimeField> FixedPoint<F> {
    /// Create a new fixed-point number from a field element and scale
    pub fn new(value: F, scale_bits: u32) -> Self {
        Self { value, scale_bits }
    }
    
    /// Convert a float to fixed-point representation
    pub fn from_f64(value: f64, scale_bits: u32) -> Self {
        let scaled = (value * (1u64 << scale_bits) as f64) as i64;
        let field_val = if scaled < 0 {
            -F::from_u64(scaled.unsigned_abs() as u64).unwrap()
        } else {
            F::from_u64(scaled as u64).unwrap()
        };
        Self::new(field_val, scale_bits)
    }
    
    /// Convert fixed-point back to float
    pub fn to_f64(&self) -> f64 {
        let scale = 1u64 << self.scale_bits;
        let value_bytes = self.value.to_repr();
        let mut value_u64 = 0u64;
        
        // Convert the field element bytes to u64
        for (i, &byte) in value_bytes.as_ref().iter().take(8).enumerate() {
            value_u64 |= (byte as u64) << (i * 8);
        }
        
        (value_u64 as f64) / (scale as f64)
    }
    
    /// Get the raw field element
    pub fn value(&self) -> F {
        self.value
    }
    
    /// Get the scale in bits
    pub fn scale_bits(&self) -> u32 {
        self.scale_bits
    }
}

impl<F: Field + FromPrimitive + Zero + PrimeField> Add for FixedPoint<F> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        assert_eq!(
            self.scale_bits, other.scale_bits,
            "Cannot add fixed-point numbers with different scales"
        );
        Self {
            value: self.value + other.value,
            scale_bits: self.scale_bits,
        }
    }
}

impl<F: Field + FromPrimitive + Zero + PrimeField> Sub for FixedPoint<F> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        assert_eq!(
            self.scale_bits, other.scale_bits,
            "Cannot subtract fixed-point numbers with different scales"
        );
        Self {
            value: self.value - other.value,
            scale_bits: self.scale_bits,
        }
    }
}

impl<F: Field + FromPrimitive + Zero + PrimeField> Mul for FixedPoint<F> {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        // When multiplying fixed-point numbers, we need to divide by the scale factor
        // to maintain the correct scale
        Self {
            value: self.value * other.value,
            scale_bits: self.scale_bits * 2,  // Scale doubles after multiplication
        }
    }
}

impl<F: Field + FromPrimitive + Zero + PrimeField> Zero for FixedPoint<F> {
    fn zero() -> Self {
        Self {
            value: F::ZERO,
            scale_bits: 32,  // Default scale
        }
    }
    
    fn is_zero(&self) -> bool {
        Field::is_zero(&self.value).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_point_creation() {
        let fp = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        assert_eq!(fp.scale_bits(), 16);
    }
    
    #[test]
    fn test_fixed_point_addition() {
        let a = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2.5, 16);
        let c = a + b;
        assert_eq!(c.scale_bits(), 16);
        
        let result = c.to_f64();
        assert!((result - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_fixed_point_subtraction() {
        let a = FixedPoint::<WrappedFp>::from_f64(3.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        let c = a - b;
        assert_eq!(c.scale_bits(), 16);
        
        let result = c.to_f64();
        assert!((result - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_fixed_point_multiplication() {
        let a = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2.0, 16);
        let c = a * b;
        assert_eq!(c.scale_bits(), 32);  // Scale doubles
        
        let result = c.to_f64();
        assert!((result - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_fixed_point_zero() {
        let zero = FixedPoint::<WrappedFp>::zero();
        assert!(zero.is_zero());
        assert_eq!(zero.to_f64(), 0.0);
    }
    
    #[test]
    fn test_fixed_point_negative() {
        let a = FixedPoint::<WrappedFp>::from_f64(-1.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2.5, 16);
        let c = a + b;
        
        let result = c.to_f64();
        assert!((result - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_fixed_point_large_numbers() {
        let a = FixedPoint::<WrappedFp>::from_f64(1000.0, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2000.0, 16);
        let c = a + b;
        
        let result = c.to_f64();
        assert!((result - 3000.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_fixed_point_small_numbers() {
        let a = FixedPoint::<WrappedFp>::from_f64(0.001, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(0.002, 16);
        let c = a + b;
        
        let result = c.to_f64();
        assert!((result - 0.003).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "Cannot add fixed-point numbers with different scales")]
    fn test_different_scales() {
        let a = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2.5, 32);
        let _c = a + b;  // Should panic
    }
    
    #[test]
    #[should_panic(expected = "Cannot subtract fixed-point numbers with different scales")]
    fn test_different_scales_subtraction() {
        let a = FixedPoint::<WrappedFp>::from_f64(1.5, 16);
        let b = FixedPoint::<WrappedFp>::from_f64(2.5, 32);
        let _c = a - b;  // Should panic
    }
} 