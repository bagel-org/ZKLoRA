use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
    poly::Rotation,
};
use num_traits::{FromPrimitive, Zero};
use crate::{quantization::FixedPoint, field_traits::WrappedFp};
use ff::PrimeField;

// Configuration for our LoRA circuit
#[derive(Clone, Debug)]
pub struct LoRAConfig {
    pub input: Column<Advice>,
    pub weight_a: Column<Advice>,
    pub weight_b: Column<Advice>,
    pub output: Column<Instance>,
}

// Our main circuit structure
#[derive(Clone)]
pub struct LoRACircuit<F: Field + FromPrimitive + Zero + PrimeField> {
    pub input: Vec<FixedPoint<F>>,
    pub weight_a: Vec<FixedPoint<F>>,
    pub weight_b: Vec<FixedPoint<F>>,
    pub _marker: std::marker::PhantomData<F>,
}

impl<F: Field + FromPrimitive + Zero + PrimeField> Circuit<F> for LoRACircuit<F> {
    type Config = LoRAConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            input: vec![],
            weight_a: vec![],
            weight_b: vec![],
            _marker: std::marker::PhantomData,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let input = meta.advice_column();
        let weight_a = meta.advice_column();
        let weight_b = meta.advice_column();
        let output = meta.instance_column();

        meta.enable_equality(input);
        meta.enable_equality(weight_a);
        meta.enable_equality(weight_b);
        meta.enable_equality(output);

        // Add constraints for matrix multiplication
        meta.create_gate("matrix_mul", |meta| {
            let input = meta.query_advice(input, Rotation::cur());
            let weight_a = meta.query_advice(weight_a, Rotation::cur());
            let weight_b = meta.query_advice(weight_b, Rotation::cur());
            let output = meta.query_instance(output, Rotation::cur());

            vec![input * weight_a * weight_b - output]
        });

        LoRAConfig {
            input,
            weight_a,
            weight_b,
            output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "lora",
            |mut region| {
                // Assign input values
                for (i, input) in self.input.iter().enumerate() {
                    region.assign_advice(
                        || format!("input {}", i),
                        config.input,
                        i,
                        || Value::known(input.value()),
                    )?;
                }

                // Assign weight matrices
                for (i, val) in self.weight_a.iter().enumerate() {
                    region.assign_advice(
                        || format!("weight_a {}", i),
                        config.weight_a,
                        i,
                        || Value::known(val.value()),
                    )?;
                }

                for (i, val) in self.weight_b.iter().enumerate() {
                    region.assign_advice(
                        || format!("weight_b {}", i),
                        config.weight_b,
                        i,
                        || Value::known(val.value()),
                    )?;
                }

                Ok(())
            },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    #[test]
    fn test_simple_lora_circuit() {
        let scale_bits = 16;
        
        // Create test inputs
        let input = vec![
            FixedPoint::<WrappedFp>::from_f64(1.0, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(2.0, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(3.0, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(4.0, scale_bits),
        ];
        
        let weight_a = vec![
            FixedPoint::<WrappedFp>::from_f64(0.1, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(0.2, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(0.3, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(0.4, scale_bits),
        ];
        
        let weight_b = vec![
            FixedPoint::<WrappedFp>::from_f64(1.0, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(1.5, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(2.0, scale_bits),
            FixedPoint::<WrappedFp>::from_f64(2.5, scale_bits),
        ];
        
        let circuit = LoRACircuit {
            input,
            weight_a,
            weight_b,
            _marker: PhantomData,
        };
        
        let prover = MockProver::run(8, &circuit, vec![vec![WrappedFp::ZERO]]).unwrap();
        assert!(prover.verify().is_ok());
    }

    #[test]
    fn test_circuit_constraints() {
        let scale_bits = 16;
        
        // Test with small values to verify constraints
        let input = vec![
            FixedPoint::<WrappedFp>::from_f64(0.5, scale_bits),
        ];
        
        let weight_a = vec![
            FixedPoint::<WrappedFp>::from_f64(2.0, scale_bits),
        ];
        
        let weight_b = vec![
            FixedPoint::<WrappedFp>::from_f64(3.0, scale_bits),
        ];
        
        let circuit = LoRACircuit {
            input,
            weight_a,
            weight_b,
            _marker: PhantomData,
        };
        
        // Expected output: 0.5 * 2.0 * 3.0 = 3.0
        let expected = FixedPoint::<WrappedFp>::from_f64(3.0, scale_bits).value();
        let prover = MockProver::run(8, &circuit, vec![vec![expected]]).unwrap();
        assert!(prover.verify().is_ok());
    }
} 