use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector},
    poly::Rotation,
    pasta::Fp,
};
use ff::Field;

use crate::quantize_to_field;

#[derive(Clone)]
pub struct LoRAConfig {
    input: Column<Advice>,
    weight_a: Column<Advice>,
    weight_b: Column<Advice>,
    output: Column<Instance>,
    selector: Selector,
}

#[derive(Default)]
pub struct LoRACircuit {
    pub input: Vec<f64>,
    pub weight_a: Vec<f64>,
    pub weight_b: Vec<f64>,
}

impl Circuit<Fp> for LoRACircuit {
    type Config = LoRAConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let input = meta.advice_column();
        let weight_a = meta.advice_column();
        let weight_b = meta.advice_column();
        let output = meta.instance_column();
        let selector = meta.selector();

        meta.enable_equality(input);
        meta.enable_equality(weight_a);
        meta.enable_equality(weight_b);
        meta.enable_equality(output);

        meta.create_gate("lora_mul", |meta| {
            let s = meta.query_selector(selector);
            let input = meta.query_advice(input, Rotation::cur());
            let weight_a = meta.query_advice(weight_a, Rotation::cur());
            let weight_b = meta.query_advice(weight_b, Rotation::cur());
            let output = meta.query_instance(output, Rotation::cur());

            vec![s * (input * weight_a * weight_b - output)]
        });

        LoRAConfig {
            input,
            weight_a,
            weight_b,
            output,
            selector,
        }
    }

    #[allow(clippy::unnecessary_lazy_evaluations)]
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let input_val = if !self.input.is_empty() {
            quantize_to_field(self.input[0])
        } else {
            Fp::zero()
        };

        let weight_a_val = if !self.weight_a.is_empty() {
            quantize_to_field(self.weight_a[0])
        } else {
            Fp::one()
        };

        let weight_b_val = if !self.weight_b.is_empty() {
            quantize_to_field(self.weight_b[0])
        } else {
            Fp::one()
        };

        let _output_val = input_val * weight_a_val * weight_b_val;

        layouter.assign_region(
            || "lora",
            |mut region| {
                config.selector.enable(&mut region, 0)?;

                region.assign_advice(
                    || "input",
                    config.input,
                    0,
                    || Value::known(input_val),
                )?;

                region.assign_advice(
                    || "weight_a",
                    config.weight_a,
                    0,
                    || Value::known(weight_a_val),
                )?;

                region.assign_advice(
                    || "weight_b",
                    config.weight_b,
                    0,
                    || Value::known(weight_b_val),
                )?;

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
    fn test_circuit_creation() {
        let circuit = LoRACircuit {
            input: vec![1.0],
            weight_a: vec![2.0],
            weight_b: vec![3.0],
        };

        // Expected output is 1.0 * 2.0 * 3.0 = 6.0
        let expected_output = vec![quantize_to_field(6.0)];
        let prover = MockProver::run(4, &circuit, vec![expected_output]).unwrap();
        assert!(prover.verify().is_ok());

        // Wrong output should fail
        let wrong_output = vec![quantize_to_field(7.0)];
        let prover = MockProver::run(4, &circuit, vec![wrong_output]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_empty_inputs() {
        let circuit = LoRACircuit {
            input: vec![],
            weight_a: vec![],
            weight_b: vec![],
        };

        // Expected output for empty inputs is 0 * 1 * 1 = 0
        let expected_output = vec![quantize_to_field(0.0)];
        let prover = MockProver::run(4, &circuit, vec![expected_output]).unwrap();
        assert!(prover.verify().is_ok());
    }

    #[test]
    fn test_negative_inputs() {
        let circuit = LoRACircuit {
            input: vec![-1.0],
            weight_a: vec![-2.0],
            weight_b: vec![-3.0],
        };

        // Expected output is abs(-1.0) * abs(-2.0) * abs(-3.0) = 6.0
        let expected_output = vec![quantize_to_field(6.0)];
        let prover = MockProver::run(4, &circuit, vec![expected_output]).unwrap();
        assert!(prover.verify().is_ok());
    }
} 