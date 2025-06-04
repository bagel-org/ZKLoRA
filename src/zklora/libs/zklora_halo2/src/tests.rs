use super::*;
use halo2_proofs::{
    arithmetic::Field,
    dev::MockProver,
    pasta::Fp,
};

#[test]
fn test_simple_matrix_multiplication() {
    // Test a simple 2x2 matrix multiplication
    let input = vec![Fp::from(1), Fp::from(2)];
    let weight_a = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];
    let weight_b = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];

    let circuit = LoRACircuit {
        input,
        weight_a,
        weight_b,
        _marker: PhantomData,
    };

    let prover = MockProver::run(8, &circuit, vec![vec![Fp::from(70)]]).unwrap();
    assert!(prover.verify().is_ok());
}

#[test]
fn test_zero_input() {
    // Test with zero inputs
    let input = vec![Fp::from(0), Fp::from(0)];
    let weight_a = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];
    let weight_b = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];

    let circuit = LoRACircuit {
        input,
        weight_a,
        weight_b,
        _marker: PhantomData,
    };

    let prover = MockProver::run(8, &circuit, vec![vec![Fp::from(0)]]).unwrap();
    assert!(prover.verify().is_ok());
}

#[test]
fn test_large_values() {
    // Test with large values that might cause overflow
    let input = vec![Fp::from(1000), Fp::from(2000)];
    let weight_a = vec![
        vec![Fp::from(1000), Fp::from(2000)],
        vec![Fp::from(3000), Fp::from(4000)],
    ];
    let weight_b = vec![
        vec![Fp::from(1000), Fp::from(2000)],
        vec![Fp::from(3000), Fp::from(4000)],
    ];

    let circuit = LoRACircuit {
        input,
        weight_a,
        weight_b,
        _marker: PhantomData,
    };

    let prover = MockProver::run(12, &circuit, vec![vec![Fp::from(70_000_000)]]).unwrap();
    assert!(prover.verify().is_ok());
}

#[test]
fn test_invalid_output() {
    // Test that the circuit rejects invalid outputs
    let input = vec![Fp::from(1), Fp::from(2)];
    let weight_a = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];
    let weight_b = vec![
        vec![Fp::from(1), Fp::from(2)],
        vec![Fp::from(3), Fp::from(4)],
    ];

    let circuit = LoRACircuit {
        input,
        weight_a,
        weight_b,
        _marker: PhantomData,
    };

    let prover = MockProver::run(8, &circuit, vec![vec![Fp::from(0)]]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_quantization() {
    // Test quantization of floating point values
    let test_values = vec![0.5, -0.5, 1.0, -1.0, 0.0];
    for val in test_values {
        let quantized = quantize_to_field::<Fp>(val);
        let dequantized = dequantize_from_field(quantized);
        assert!((val - dequantized).abs() < 1e-6);
    }
}

fn quantize_to_field<F: Field>(value: f64) -> F {
    // TODO: Implement proper quantization
    unimplemented!()
}

fn dequantize_from_field<F: Field>(value: F) -> f64 {
    // TODO: Implement proper dequantization
    unimplemented!()
} 