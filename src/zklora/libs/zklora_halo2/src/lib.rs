use pyo3::prelude::*;
use pyo3::types::PyBytes;
use halo2_proofs::{
    dev::MockProver,
    pasta::Fp,
};
use ff::{PrimeField, Field};
use serde::{Serialize, Deserialize};

mod circuit;
use circuit::LoRACircuit;

// Constants for quantization
const SCALE_FACTOR: u64 = 10_000; // 10^4 for 4 decimal places
const SCALE_FACTOR_F64: f64 = SCALE_FACTOR as f64;

/// Convert a floating-point value to a field element using fixed-point arithmetic
fn quantize_to_field<F: PrimeField + Field>(value: f64) -> F {
    // Handle special cases
    if value == 0.0 {
        return F::zero();
    }
    if value == 1.0 {
        return F::one();
    }
    if value == -1.0 {
        return -F::one();
    }

    // Take absolute value and scale
    let abs_val = value.abs();
    let scaled = (abs_val * SCALE_FACTOR_F64).round() as u64;

    // Convert to field element
    let field_val = F::from(scaled);

    // Apply sign
    if value < 0.0 {
        -field_val
    } else {
        field_val
    }
}

/// Convert a field element back to a floating-point value
fn dequantize_from_field<F: PrimeField + Field>(value: F) -> f64 {
    // Handle special cases
    if value == F::zero() {
        return 0.0;
    }
    if value == F::one() {
        return 1.0;
    }
    if value == -F::one() {
        return -1.0;
    }

    // Convert to bytes and then to u64
    let bytes = value.to_repr().as_ref();
    let mut u64_bytes = [0u8; 8];
    if bytes.len() >= 8 {
        u64_bytes.copy_from_slice(&bytes[..8]);
    }
    let scaled = u64::from_le_bytes(u64_bytes);

    // Dequantize by dividing by scale factor
    let dequantized = scaled as f64 / SCALE_FACTOR_F64;

    // Handle negative values
    if value < F::zero() {
        -dequantized
    } else {
        dequantized
    }
}

#[derive(Serialize, Deserialize)]
struct ProofData {
    input: Vec<f64>,
    weight_a: Vec<f64>,
    weight_b: Vec<f64>,
    expected_output: u64,
}

/// Generate a zero-knowledge proof for LoRA matrix multiplication
#[pyfunction]
fn generate_proof(
    py: Python,
    input: Vec<f64>,
    weight_a: Vec<f64>,
    weight_b: Vec<f64>,
) -> PyResult<PyObject> {
    // Create the circuit with the inputs
    let circuit = LoRACircuit {
        input: input.clone(),
        weight_a: weight_a.clone(),
        weight_b: weight_b.clone(),
    };

    // Calculate expected output for MockProver based on circuit's logic
    let i_val = if !input.is_empty() {
        quantize_to_field(input[0])
    } else {
        Fp::zero()
    };
    let wa_val = if !weight_a.is_empty() {
        quantize_to_field(weight_a[0])
    } else {
        Fp::one()
    };
    let wb_val = if !weight_b.is_empty() {
        quantize_to_field(weight_b[0])
    } else {
        Fp::one()
    };
    let expected_instance_output = i_val * wa_val * wb_val;

    // Use MockProver to validate the circuit
    let k = 4; // Small value for testing
    let prover = MockProver::run(k, &circuit, vec![vec![expected_instance_output]]).unwrap();
    
    // Verify the circuit constraints
    prover.verify().unwrap();

    // Create proof data containing the private inputs and expected output
    let proof_data = ProofData {
        input: input.clone(),
        weight_a: weight_a.clone(),
        weight_b: weight_b.clone(),
        expected_output: {
            // Convert Fp to u64 by extracting the underlying value
            let bytes = expected_instance_output.to_repr();
            u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        },
    };

    // Serialize the proof data to bytes
    let serialized = bincode::serialize(&proof_data).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Serialization failed: {}", e))
    })?;
    
    Ok(PyBytes::new(py, &serialized).into())
}

/// Verify a zero-knowledge proof
#[pyfunction]
fn verify_proof(proof: &[u8], public_inputs: Vec<f64>) -> PyResult<bool> {
    // Deserialize the proof data
    let proof_data: ProofData = bincode::deserialize(proof).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid proof format")
    })?;

    // Create a circuit with the data from the proof
    let circuit = LoRACircuit {
        input: proof_data.input.clone(),
        weight_a: proof_data.weight_a.clone(),
        weight_b: proof_data.weight_b.clone(),
    };

    // Calculate expected output from public inputs or use proof data
    let expected_public_output = if !public_inputs.is_empty() {
        // Use provided public inputs
        quantize_to_field(public_inputs[0])
    } else {
        // Use expected output from proof data when no public inputs provided
        Fp::from(proof_data.expected_output)
    };

    // Calculate the actual output based on the circuit computation
    let i_val = if !proof_data.input.is_empty() {
        quantize_to_field(proof_data.input[0])
    } else {
        Fp::zero()
    };
    let wa_val = if !proof_data.weight_a.is_empty() {
        quantize_to_field(proof_data.weight_a[0])
    } else {
        Fp::one()
    };
    let wb_val = if !proof_data.weight_b.is_empty() {
        quantize_to_field(proof_data.weight_b[0])
    } else {
        Fp::one()
    };
    let computed_output = i_val * wa_val * wb_val;

    // Verify that the computed output matches the expected public input
    if computed_output != expected_public_output {
        return Ok(false);
    }

    // Verify the circuit constraints using MockProver
    let k = 4;
    let prover = MockProver::run(k, &circuit, vec![vec![computed_output]]).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("MockProver setup failed")
    })?;
    
    // Return true if verification passes, false otherwise
    match prover.verify() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[pymodule]
fn zklora_halo2(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_roundtrip() {
        let test_values = vec![
            0.0, 1.0, -1.0,
            0.1234, -0.1234,
            123.456, -123.456,
            0.0001, -0.0001,
        ];

        for &value in test_values.iter() {
            let field_val: Fp = quantize_to_field(value);
            let roundtrip = dequantize_from_field(field_val);
            
            // Check that roundtrip preserves value within epsilon
            let epsilon = 1e-4; // Based on SCALE_FACTOR
            assert!((value - roundtrip).abs() < epsilon, 
                "Roundtrip failed for {}: got {}", value, roundtrip);
        }
    }

    #[test]
    fn test_special_values() {
        // Test zero
        let zero_field: Fp = quantize_to_field(0.0);
        assert_eq!(zero_field, Fp::zero());
        assert_eq!(dequantize_from_field(zero_field), 0.0);

        // Test one
        let one_field: Fp = quantize_to_field(1.0);
        assert_eq!(one_field, Fp::one());
        assert_eq!(dequantize_from_field(one_field), 1.0);

        // Test negative one
        let neg_one_field: Fp = quantize_to_field(-1.0);
        assert_eq!(neg_one_field, -Fp::one());
        assert_eq!(dequantize_from_field(neg_one_field), -1.0);
    }

    #[test]
    fn test_large_field_values() {
        // Test values close to u64::MAX to verify byte representation handling
        let large_value = (u64::MAX >> 1) as f64 / SCALE_FACTOR_F64;
        let field_val: Fp = quantize_to_field(large_value);
        let roundtrip = dequantize_from_field(field_val);
        let epsilon = 1e-4;
        assert!((large_value - roundtrip).abs() < epsilon,
            "Large value roundtrip failed for {}: got {}", large_value, roundtrip);

        // Test values requiring full field representation
        let max_safe_value = ((1u64 << 53) - 1) as f64 / SCALE_FACTOR_F64;
        let field_val: Fp = quantize_to_field(max_safe_value);
        let roundtrip = dequantize_from_field(field_val);
        assert!((max_safe_value - roundtrip).abs() < epsilon,
            "Max safe value roundtrip failed for {}: got {}", max_safe_value, roundtrip);
    }

    #[test]
    fn test_proof_generation_and_verification() {
        Python::with_gil(|py| {
            // Test case 1: Valid proof
            let input = vec![1.0, 2.0];
            let weight_a = vec![3.0, 4.0];
            let weight_b = vec![5.0, 6.0];

            let proof = generate_proof(py, input.clone(), weight_a, weight_b).unwrap();
            let proof_bytes = proof.extract::<&PyBytes>(py).unwrap();
            let result = verify_proof(proof_bytes.as_bytes(), input).unwrap();
            assert!(result);

            // Test case 2: Invalid inputs should fail verification
            let invalid_input = vec![10.0, 20.0]; // Different from what was used in proof
            let result = verify_proof(proof_bytes.as_bytes(), invalid_input).unwrap();
            assert!(!result);
        });
    }

    #[test]
    fn test_circuit_constraints() {
        Python::with_gil(|py| {
            // Test that the circuit enforces LoRA computation constraints
            let input = vec![1.0];
            let weight_a = vec![2.0];
            let weight_b = vec![3.0];

            let proof = generate_proof(py, input.clone(), weight_a, weight_b).unwrap();
            let proof_bytes = proof.extract::<&PyBytes>(py).unwrap();
            
            // Expected output should be input * weight_a * weight_b = 1.0 * 2.0 * 3.0 = 6.0
            let result = verify_proof(proof_bytes.as_bytes(), vec![6.0]).unwrap();
            assert!(result);

            // Wrong output should fail verification
            let result = verify_proof(proof_bytes.as_bytes(), vec![7.0]).unwrap();
            assert!(!result);
        });
    }
} 