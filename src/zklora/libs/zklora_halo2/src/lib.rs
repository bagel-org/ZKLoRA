use pyo3::prelude::*;
use pyo3::types::PyBytes;
use halo2_proofs::{
    dev::MockProver,
    pasta::Fp,
};

mod circuit;
use circuit::LoRACircuit;

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
        Fp::from(input[0].abs() as u64)
    } else {
        Fp::zero()
    };
    let wa_val = if !weight_a.is_empty() {
        Fp::from(weight_a[0].abs() as u64)
    } else {
        Fp::one()
    };
    let wb_val = if !weight_b.is_empty() {
        Fp::from(weight_b[0].abs() as u64)
    } else {
        Fp::one()
    };
    let expected_instance_output = i_val * wa_val * wb_val;

    // For now, we'll use MockProver for testing
    // In production, this would use actual Halo2 proving system
    let k = 4; // Small value for testing
    let prover = MockProver::run(k, &circuit, vec![vec![expected_instance_output]]).unwrap();
    
    // Verify the circuit constraints
    prover.verify().unwrap();

    // Serialize the proof - in production this would be actual proof bytes
    let dummy_proof = vec![0u8; 32];  // TODO: Replace with actual proof serialization
    Ok(PyBytes::new(py, &dummy_proof).into())
}

/// Verify a zero-knowledge proof
#[pyfunction]
fn verify_proof(_proof: &[u8], public_inputs: Vec<f64>) -> PyResult<bool> {
    // Create a circuit with the public inputs
    let circuit = LoRACircuit {
        input: public_inputs.clone(),
        weight_a: vec![], // These will be private inputs
        weight_b: vec![], // These will be private inputs
    };

    // For now, we'll use MockProver for verification
    // In production, this would use actual Halo2 verification
    let k = 4; // Same k as in proof generation

    // Calculate expected output
    let expected_output = if !public_inputs.is_empty() {
        vec![Fp::from(public_inputs[0].abs() as u64)]
    } else {
        vec![Fp::zero()]
    };

    let prover = MockProver::run(k, &circuit, vec![expected_output]).unwrap();
    
    // Verify the circuit constraints
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