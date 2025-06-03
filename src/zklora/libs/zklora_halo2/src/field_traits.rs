use halo2_proofs::{arithmetic::Field, pasta::Fp};
use num_traits::{FromPrimitive, Zero, One};
use std::ops::{Add, Mul, Sub, Neg, AddAssign, SubAssign, MulAssign};
use std::iter::{Sum, Product};
use std::cmp::Ordering;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};
use ff::PrimeField;

/// A wrapper around Fp to implement external traits
#[derive(Clone, Debug, PartialEq, Eq, Copy, Default)]
pub struct WrappedFp(pub Fp);

impl PartialOrd for WrappedFp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WrappedFp {
    fn cmp(&self, other: &Self) -> Ordering {
        // Convert to bytes and compare lexicographically
        let self_bytes = self.0.to_repr();
        let other_bytes = other.0.to_repr();
        self_bytes.as_ref().cmp(other_bytes.as_ref())
    }
}

impl FromPrimitive for WrappedFp {
    fn from_i64(n: i64) -> Option<Self> {
        if n < 0 {
            Some(WrappedFp(-Fp::from(n.unsigned_abs() as u64)))
        } else {
            Some(WrappedFp(Fp::from(n as u64)))
        }
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(WrappedFp(Fp::from(n)))
    }

    fn from_f64(n: f64) -> Option<Self> {
        let scaled = (n * (1u64 << 32) as f64) as i64;
        if scaled < 0 {
            Some(WrappedFp(-Fp::from(scaled.unsigned_abs() as u64)))
        } else {
            Some(WrappedFp(Fp::from(scaled as u64)))
        }
    }
}

impl Zero for WrappedFp {
    fn zero() -> Self {
        WrappedFp(Fp::zero())
    }

    fn is_zero(&self) -> bool {
        Field::is_zero(&self.0).into()
    }
}

impl One for WrappedFp {
    fn one() -> Self {
        WrappedFp(Fp::one())
    }
}

impl Add for WrappedFp {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        WrappedFp(self.0 + other.0)
    }
}

impl<'a> Add<&'a WrappedFp> for WrappedFp {
    type Output = Self;

    fn add(self, other: &'a WrappedFp) -> Self {
        WrappedFp(self.0 + other.0)
    }
}

impl Sub for WrappedFp {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        WrappedFp(self.0 - other.0)
    }
}

impl<'a> Sub<&'a WrappedFp> for WrappedFp {
    type Output = Self;

    fn sub(self, other: &'a WrappedFp) -> Self {
        WrappedFp(self.0 - other.0)
    }
}

impl Mul for WrappedFp {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        WrappedFp(self.0 * other.0)
    }
}

impl<'a> Mul<&'a WrappedFp> for WrappedFp {
    type Output = Self;

    fn mul(self, other: &'a WrappedFp) -> Self {
        WrappedFp(self.0 * other.0)
    }
}

impl Neg for WrappedFp {
    type Output = Self;

    fn neg(self) -> Self {
        WrappedFp(-self.0)
    }
}

impl AddAssign for WrappedFp {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl<'a> AddAssign<&'a WrappedFp> for WrappedFp {
    fn add_assign(&mut self, other: &'a WrappedFp) {
        self.0 += other.0;
    }
}

impl SubAssign for WrappedFp {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl<'a> SubAssign<&'a WrappedFp> for WrappedFp {
    fn sub_assign(&mut self, other: &'a WrappedFp) {
        self.0 -= other.0;
    }
}

impl MulAssign for WrappedFp {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl<'a> MulAssign<&'a WrappedFp> for WrappedFp {
    fn mul_assign(&mut self, other: &'a WrappedFp) {
        self.0 *= other.0;
    }
}

impl Sum for WrappedFp {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a WrappedFp> for WrappedFp {
    fn sum<I: Iterator<Item = &'a WrappedFp>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl Product for WrappedFp {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> Product<&'a WrappedFp> for WrappedFp {
    fn product<I: Iterator<Item = &'a WrappedFp>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl ConditionallySelectable for WrappedFp {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        WrappedFp(Fp::conditional_select(&a.0, &b.0, choice))
    }
}

impl ConstantTimeEq for WrappedFp {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl Field for WrappedFp {
    const ZERO: Self = WrappedFp(Fp::ZERO);
    const ONE: Self = WrappedFp(Fp::ONE);

    fn random(mut rng: impl rand_core::RngCore) -> Self {
        WrappedFp(Fp::random(&mut rng))
    }

    fn square(&self) -> Self {
        WrappedFp(self.0.square())
    }

    fn double(&self) -> Self {
        WrappedFp(self.0.double())
    }

    fn invert(&self) -> CtOption<Self> {
        self.0.invert().map(WrappedFp)
    }

    fn sqrt(&self) -> CtOption<Self> {
        self.0.sqrt().map(WrappedFp)
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        let (choice, result) = Fp::sqrt_ratio(&num.0, &div.0);
        (choice, WrappedFp(result))
    }
}

impl From<WrappedFp> for Fp {
    fn from(w: WrappedFp) -> Self {
        w.0
    }
}

impl From<Fp> for WrappedFp {
    fn from(f: Fp) -> Self {
        WrappedFp(f)
    }
}

impl PrimeField for WrappedFp {
    type Repr = <Fp as PrimeField>::Repr;
    const S: u32 = Fp::S;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        Fp::from_repr(repr).map(WrappedFp)
    }

    fn to_repr(&self) -> Self::Repr {
        self.0.to_repr()
    }

    fn is_odd(&self) -> Choice {
        self.0.is_odd()
    }

    const MODULUS: &'static str = Fp::MODULUS;
    const NUM_BITS: u32 = Fp::NUM_BITS;
    const CAPACITY: u32 = Fp::CAPACITY;
    const TWO_INV: Self = WrappedFp(Fp::TWO_INV);
    const MULTIPLICATIVE_GENERATOR: Self = WrappedFp(Fp::MULTIPLICATIVE_GENERATOR);
    const ROOT_OF_UNITY: Self = WrappedFp(Fp::ROOT_OF_UNITY);
    const ROOT_OF_UNITY_INV: Self = WrappedFp(Fp::ROOT_OF_UNITY_INV);
    const DELTA: Self = WrappedFp(Fp::DELTA);
}

impl From<u64> for WrappedFp {
    fn from(value: u64) -> Self {
        WrappedFp(Fp::from(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrapped_fp_arithmetic() {
        let a = WrappedFp::from_f64(1.5).unwrap();
        let b = WrappedFp::from_f64(2.5).unwrap();
        
        let c = a + b;
        assert!(!Field::is_zero(&c));
        
        let d = c - b;
        assert_eq!(d.0, a.0);
        
        let e = a * b;
        assert!(!Field::is_zero(&e));
        
        let f = -a;
        assert_eq!(f + a, WrappedFp::ZERO);
    }

    #[test]
    fn test_wrapped_fp_conversions() {
        let a = WrappedFp::from_i64(-5).unwrap();
        let b = WrappedFp::from_u64(5).unwrap();
        let c = WrappedFp::from_f64(-5.0).unwrap();
        
        assert_eq!(-a.0, b.0);
        assert_eq!(a.0, c.0);
    }

    #[test]
    fn test_wrapped_fp_zero() {
        let zero = WrappedFp::ZERO;
        assert!(Field::is_zero(&zero));
        
        let a = WrappedFp::from_f64(1.5).unwrap();
        assert!(!Field::is_zero(&a));
        assert_eq!(a + zero, a);
        assert_eq!(zero + a, a);
    }

    #[test]
    fn test_wrapped_fp_field_ops() {
        let a = WrappedFp::from_f64(2.0).unwrap();
        let b = WrappedFp::from_f64(4.0).unwrap();
        
        let square = a.square();
        assert_eq!(square, a * a);
        
        let double = a.double();
        assert_eq!(double, a + a);
        
        let inv = a.invert().unwrap();
        assert_eq!(a * inv, WrappedFp::ONE);
        
        let (_, sqrt) = WrappedFp::sqrt_ratio(&b, &a);
        assert_eq!(sqrt.square() * a, b);
    }

    #[test]
    fn test_wrapped_fp_ref_ops() {
        let mut a = WrappedFp::from_f64(2.0).unwrap();
        let b = WrappedFp::from_f64(3.0).unwrap();
        
        a += &b;
        assert_eq!(a, WrappedFp::from_f64(5.0).unwrap());
        
        a -= &b;
        assert_eq!(a, WrappedFp::from_f64(2.0).unwrap());
        
        a *= &b;
        assert_eq!(a, WrappedFp::from_f64(6.0).unwrap());
    }

    #[test]
    fn test_wrapped_fp_sum_product() {
        let nums = vec![
            WrappedFp::from_f64(1.0).unwrap(),
            WrappedFp::from_f64(2.0).unwrap(),
            WrappedFp::from_f64(3.0).unwrap(),
        ];
        
        let sum: WrappedFp = nums.iter().sum();
        assert_eq!(sum, WrappedFp::from_f64(6.0).unwrap());
        
        let product: WrappedFp = nums.iter().product();
        assert_eq!(product, WrappedFp::from_f64(6.0).unwrap());
    }

    #[test]
    fn test_wrapped_fp_ordering() {
        let a = WrappedFp::from_f64(1.0).unwrap();
        let b = WrappedFp::from_f64(2.0).unwrap();
        let c = WrappedFp::from_f64(1.0).unwrap();
        
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
    }
} 