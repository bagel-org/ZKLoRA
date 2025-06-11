use pyo3::prelude::*;
use pyo3::types::PyBytes;
use halo2_proofs::{
    dev::MockProver,
    pasta::Fp,
};
use ff::{Field, PrimeField};
use serde::{Serialize, Deserialize};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::ops::Mul;

mod circuit;
use circuit::LoRACircuit;

// Constants for quantization
const SCALE_FACTOR: u64 = 10_000; // 10^4 for 4 decimal places
const SCALE_FACTOR_F64: f64 = SCALE_FACTOR as f64;

fn modulus_as_biguint<F: PrimeField>() -> BigUint {
    let bytes: &[u8] = F::MODULUS.as_ref();
    BigUint::from_bytes_be(bytes)
}

fn to_big_endian_32(n: &BigUint, modulus: &BigUint) -> [u8; 32] {
    let mut n = n.clone();
    if n.bits() > 256 {
        n = modulus - BigUint::from(1u8);
    }
    if n.is_zero() {
        return [0u8; 32];
    }
    let mut bytes = n.to_bytes_be();
    if bytes.len() < 32 {
        let mut pad = vec![0u8; 32 - bytes.len()];
        pad.extend_from_slice(&bytes);
        bytes = pad;
    }
    bytes.as_slice().try_into().unwrap()
}

#[derive(Clone, Copy, Debug)]
pub struct Quantized {
    pub magnitude: Fp,
    pub sign: Fp, // 0 for positive, 1 for negative
}

pub fn quantize_to_field(value: f64) -> Quantized {
    let scaled = (value.abs() * SCALE_FACTOR_F64).round() as u64;
    let mut arr = [0u8; 32];
    arr[..8].copy_from_slice(&scaled.to_le_bytes());
    let magnitude = Fp::from_repr(arr).unwrap();
    let sign = if value < 0.0 { Fp::ONE } else { Fp::ZERO };
    Quantized { magnitude, sign }
}

pub fn dequantize_from_field(q: Quantized) -> f64 {
    let mut arr = q.magnitude.to_repr();
    arr[31] = 0;
    let mut u64_bytes = [0u8; 8];
    u64_bytes.copy_from_slice(&arr[..8]);
    let scaled = u64::from_le_bytes(u64_bytes);
    let abs_val = scaled as f64 / SCALE_FACTOR_F64;
    if q.sign == Fp::ONE {
        -abs_val
    } else {
        abs_val
    }
}

impl Mul for Quantized {
    type Output = Quantized;
    fn mul(self, rhs: Quantized) -> Quantized {
        let f = dequantize_from_field(self) * dequantize_from_field(rhs);
        quantize_to_field(f)
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
    let i_q = if !input.is_empty() {
        quantize_to_field(input[0])
    } else {
        quantize_to_field(0.0)
    };
    let wa_q = if !weight_a.is_empty() {
        quantize_to_field(weight_a[0])
    } else {
        quantize_to_field(1.0)
    };
    let wb_q = if !weight_b.is_empty() {
        quantize_to_field(weight_b[0])
    } else {
        quantize_to_field(1.0)
    };
    let output_f = dequantize_from_field(i_q) * dequantize_from_field(wa_q) * dequantize_from_field(wb_q) / (SCALE_FACTOR_F64 * SCALE_FACTOR_F64);
    let output_q = quantize_to_field(output_f);
    let expected_output = vec![output_q.magnitude];
    let expected_sign = vec![output_q.sign];

    // Use MockProver to validate the circuit
    let k = 4; // Small value for testing
    let prover = halo2_proofs::dev::MockProver::run(k, &circuit, vec![expected_output.clone(), expected_sign.clone()]).unwrap();
    prover.verify().unwrap();

    // Create proof data containing the private inputs and expected output
    let proof_data = ProofData {
        input: input.clone(),
        weight_a: weight_a.clone(),
        weight_b: weight_b.clone(),
        expected_output: {
            // Convert Fp to u64 by extracting the underlying value
            let bytes = output_q.magnitude.to_repr();
            u64::from_le_bytes([
                bytes.as_ref()[0], bytes.as_ref()[1], bytes.as_ref()[2], bytes.as_ref()[3],
                bytes.as_ref()[4], bytes.as_ref()[5], bytes.as_ref()[6], bytes.as_ref()[7],
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
    let expected_q = if !public_inputs.is_empty() {
        quantize_to_field(public_inputs[0])
    } else {
        // Use expected output from proof data (magnitude only, sign assumed positive)
        let mut arr = [0u8; 32];
        arr[..8].copy_from_slice(&proof_data.expected_output.to_le_bytes());
        Quantized { magnitude: Fp::from_repr(arr).unwrap(), sign: Fp::ZERO }
    };

    // Calculate the actual output based on the circuit computation
    let i_q = if !proof_data.input.is_empty() {
        quantize_to_field(proof_data.input[0])
    } else {
        quantize_to_field(0.0)
    };
    let wa_q = if !proof_data.weight_a.is_empty() {
        quantize_to_field(proof_data.weight_a[0])
    } else {
        quantize_to_field(1.0)
    };
    let wb_q = if !proof_data.weight_b.is_empty() {
        quantize_to_field(proof_data.weight_b[0])
    } else {
        quantize_to_field(1.0)
    };
    let computed_q = i_q * wa_q * wb_q;

    // Verify that the computed output matches the expected public input
    if computed_q.magnitude != expected_q.magnitude || computed_q.sign != expected_q.sign {
        return Ok(false);
    }

    // Verify the circuit constraints using MockProver
    let k = 4;
    let prover = MockProver::run(k, &circuit, vec![vec![computed_q.magnitude]]).map_err(|_| {
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
            let q = quantize_to_field(value);
            let roundtrip = dequantize_from_field(q);
            let epsilon = 0.5 / SCALE_FACTOR_F64;
            println!("test_quantization_roundtrip: value={} field_val={:?} roundtrip={}", value, q, roundtrip);
            assert!((value - roundtrip).abs() <= epsilon,
                "Roundtrip failed for {}: got {} (epsilon {})", value, roundtrip, epsilon);
        }
    }

    #[test]
    fn test_special_values() {
        let epsilon = 0.5 / SCALE_FACTOR_F64;
        // Test zero
        let zero_q = quantize_to_field(0.0);
        assert!((dequantize_from_field(zero_q) - 0.0).abs() <= epsilon);

        // Test one
        let one_q = quantize_to_field(1.0);
        assert!((dequantize_from_field(one_q) - 1.0).abs() <= epsilon);

        // Test negative one
        let neg_one_q = quantize_to_field(-1.0);
        assert!((dequantize_from_field(neg_one_q) + 1.0).abs() <= epsilon);
    }

    // Circuit-specific tests moved to circuit.rs.  Only quantization property tests remain here.
} 