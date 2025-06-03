use pyo3::prelude::*;
use std::marker::PhantomData;

mod circuit;
mod quantization;
mod field_traits;

pub use circuit::{LoRACircuit, LoRAConfig};
pub use quantization::FixedPoint;
pub use field_traits::WrappedFp;

#[pymodule]
fn zklora_halo2(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfunction]
    fn generate_proof(
        input: Vec<f64>,
        weight_a: Vec<f64>,
        weight_b: Vec<f64>,
    ) -> PyResult<Vec<u8>> {
        // Convert inputs to fixed-point
        let scale_bits = 16;

        let input_fixed: Vec<FixedPoint<WrappedFp>> = input.iter()
            .map(|&x| FixedPoint::<WrappedFp>::from_f64(x, scale_bits))
            .collect();
            
        let weight_a_fixed: Vec<FixedPoint<WrappedFp>> = weight_a.iter()
            .map(|&x| FixedPoint::<WrappedFp>::from_f64(x, scale_bits))
            .collect();
            
        let weight_b_fixed: Vec<FixedPoint<WrappedFp>> = weight_b.iter()
            .map(|&x| FixedPoint::<WrappedFp>::from_f64(x, scale_bits))
            .collect();
            
        let _circuit = LoRACircuit {
            input: input_fixed,
            weight_a: weight_a_fixed,
            weight_b: weight_b_fixed,
            _marker: PhantomData,
        };
        
        // TODO: Implement actual proof generation
        Ok(vec![0; 32])  // Return dummy proof for now
    }

    #[pyfunction]
    fn verify_proof(_proof: Vec<u8>, _public_inputs: Vec<f64>) -> PyResult<bool> {
        // TODO: Implement actual proof verification
        Ok(true)  // Return dummy result for now
    }

    m.add_function(wrap_pyfunction!(generate_proof, m)?)?;
    m.add_function(wrap_pyfunction!(verify_proof, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_bindings() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight_a = vec![0.1, 0.2, 0.3, 0.4];
        let weight_b = vec![1.0, 1.5, 2.0, 2.5];

        let proof = generate_proof(input, weight_a, weight_b).unwrap();
        assert_eq!(proof.len(), 32);

        let result = verify_proof(proof, vec![1.0, 2.0]).unwrap();
        assert!(result);
    }
} 