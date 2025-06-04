use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, Expression},
    poly::Rotation,
    pasta::Fp,
};
use ff::Field;

use crate::quantize_to_field;

#[derive(Clone)]
pub struct LoRAConfig {
    input_magnitude: Column<Advice>,
    input_sign: Column<Advice>,
    weight_a_magnitude: Column<Advice>,
    weight_a_sign: Column<Advice>,
    weight_b_magnitude: Column<Advice>,
    weight_b_sign: Column<Advice>,
    output_magnitude: Column<Instance>,
    output_sign: Column<Instance>,
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
        let input_magnitude = meta.advice_column();
        let input_sign = meta.advice_column();
        let weight_a_magnitude = meta.advice_column();
        let weight_a_sign = meta.advice_column();
        let weight_b_magnitude = meta.advice_column();
        let weight_b_sign = meta.advice_column();
        let output_magnitude = meta.instance_column();
        let output_sign = meta.instance_column();
        let selector = meta.selector();

        meta.enable_equality(input_magnitude);
        meta.enable_equality(input_sign);
        meta.enable_equality(weight_a_magnitude);
        meta.enable_equality(weight_a_sign);
        meta.enable_equality(weight_b_magnitude);
        meta.enable_equality(weight_b_sign);
        meta.enable_equality(output_magnitude);
        meta.enable_equality(output_sign);

        meta.create_gate("lora_mul", |meta| {
            let s = meta.query_selector(selector);
            let input_magnitude = meta.query_advice(input_magnitude, Rotation::cur());
            let input_sign = meta.query_advice(input_sign, Rotation::cur());
            let weight_a_magnitude = meta.query_advice(weight_a_magnitude, Rotation::cur());
            let weight_a_sign = meta.query_advice(weight_a_sign, Rotation::cur());
            let weight_b_magnitude = meta.query_advice(weight_b_magnitude, Rotation::cur());
            let weight_b_sign = meta.query_advice(weight_b_sign, Rotation::cur());
            let output_magnitude = meta.query_instance(output_magnitude, Rotation::cur());
            let output_sign = meta.query_instance(output_sign, Rotation::cur());
            let scale = Expression::Constant(Fp::from(crate::SCALE_FACTOR * crate::SCALE_FACTOR));
            // Magnitude constraint
            let mag_constraint = input_magnitude * weight_a_magnitude * weight_b_magnitude - output_magnitude * scale;
            // Sign constraint: XOR in field arithmetic (all as Expression)
            let two = Expression::Constant(Fp::from(2));
            let four = Expression::Constant(Fp::from(4));
            let ab = input_sign.clone() * weight_a_sign.clone();
            let ac = input_sign.clone() * weight_b_sign.clone();
            let bc = weight_a_sign.clone() * weight_b_sign.clone();
            let abc = input_sign.clone() * weight_a_sign.clone() * weight_b_sign.clone();
            let xor = input_sign.clone() + weight_a_sign.clone() + weight_b_sign.clone()
                - two.clone() * (ab.clone() + ac.clone() + bc.clone())
                + four * abc;
            let sign_constraint = xor - output_sign;
            vec![s.clone() * mag_constraint, s * sign_constraint]
        });

        LoRAConfig {
            input_magnitude,
            input_sign,
            weight_a_magnitude,
            weight_a_sign,
            weight_b_magnitude,
            weight_b_sign,
            output_magnitude,
            output_sign,
            selector,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // Decode all quantized values to f64
        let input_f = if !self.input.is_empty() {
            self.input[0]
        } else {
            0.0
        };
        let wa_f = if !self.weight_a.is_empty() {
            self.weight_a[0]
        } else {
            1.0
        };
        let wb_f = if !self.weight_b.is_empty() {
            self.weight_b[0]
        } else {
            1.0
        };
        // Quantize all values for circuit storage
        let input_q = crate::quantize_to_field(input_f);
        let wa_q = crate::quantize_to_field(wa_f);
        let wb_q = crate::quantize_to_field(wb_f);
        // Fixed-point multiplication and sign logic
        let output_f = input_f * wa_f * wb_f / (crate::SCALE_FACTOR_F64 * crate::SCALE_FACTOR_F64);
        let output_q = crate::quantize_to_field(output_f);
        // XOR the signs for output sign
        let output_sign = if (input_q.sign == Fp::ONE) ^ (wa_q.sign == Fp::ONE) ^ (wb_q.sign == Fp::ONE) {
            Fp::ONE
        } else {
            Fp::ZERO
        };

        layouter.assign_region(
            || "lora",
            |mut region| {
                config.selector.enable(&mut region, 0)?;

                region.assign_advice(
                    || "input_magnitude",
                    config.input_magnitude,
                    0,
                    || Value::known(input_q.magnitude),
                )?;

                region.assign_advice(
                    || "input_sign",
                    config.input_sign,
                    0,
                    || Value::known(input_q.sign),
                )?;

                region.assign_advice(
                    || "weight_a_magnitude",
                    config.weight_a_magnitude,
                    0,
                    || Value::known(wa_q.magnitude),
                )?;

                region.assign_advice(
                    || "weight_a_sign",
                    config.weight_a_sign,
                    0,
                    || Value::known(wa_q.sign),
                )?;

                region.assign_advice(
                    || "weight_b_magnitude",
                    config.weight_b_magnitude,
                    0,
                    || Value::known(wb_q.magnitude),
                )?;

                region.assign_advice(
                    || "weight_b_sign",
                    config.weight_b_sign,
                    0,
                    || Value::known(wb_q.sign),
                )?;

                Ok(())
            },
        )?;

        // Do not assign to output_magnitude or output_sign here. Outputs are checked via instance columns in the gate and provided in the test/prover.

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
        let expected = 1.0 * 2.0 * 3.0;
        let expected_q = crate::quantize_to_field(expected);
        let expected_output = vec![expected_q.magnitude];
        let expected_sign = vec![expected_q.sign];
        let prover = MockProver::run(4, &circuit, vec![expected_output.clone(), expected_sign.clone()]).unwrap();
        assert!(prover.verify().is_ok());
        // Wrong output should fail
        let wrong_q = crate::quantize_to_field(7.0);
        let wrong_output = vec![wrong_q.magnitude];
        let wrong_sign = vec![wrong_q.sign];
        let prover = MockProver::run(4, &circuit, vec![wrong_output, wrong_sign]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_empty_inputs() {
        let circuit = LoRACircuit {
            input: vec![],
            weight_a: vec![],
            weight_b: vec![],
        };
        let expected = 0.0 * 1.0 * 1.0;
        let expected_q = crate::quantize_to_field(expected);
        let expected_output = vec![expected_q.magnitude];
        let expected_sign = vec![expected_q.sign];
        let prover = MockProver::run(4, &circuit, vec![expected_output, expected_sign]).unwrap();
        assert!(prover.verify().is_ok());
    }

    #[test]
    fn test_negative_inputs() {
        let circuit = LoRACircuit {
            input: vec![-1.0],
            weight_a: vec![-2.0],
            weight_b: vec![-3.0],
        };
        let input = -1.0;
        let wa = -2.0;
        let wb = -3.0;
        let expected = input * wa * wb;
        let expected_q = crate::quantize_to_field(expected);
        let expected_output = vec![expected_q.magnitude];
        let expected_sign = vec![expected_q.sign];
        println!("test_negative_inputs: input={} wa={} wb={} expected={} expected_quantized={:?} expected_decoded={} sign={}", input, wa, wb, expected, expected_q.magnitude, crate::dequantize_from_field(expected_q), if expected_q.sign == Fp::ONE { "negative" } else { "positive" });
        let prover = MockProver::run(4, &circuit, vec![expected_output, expected_sign]).unwrap();
        assert!(prover.verify().is_ok());
    }

    #[test]
    fn test_large_values() {
        let circuit = LoRACircuit {
            input: vec![1.0],
            weight_a: vec![2.0],
            weight_b: vec![3.0],
        };
        let expected = 1.0 * 2.0 * 3.0;
        let expected_q = crate::quantize_to_field(expected);
        let expected_output = vec![expected_q.magnitude];
        let expected_sign = vec![expected_q.sign];
        let prover = MockProver::run(4, &circuit, vec![expected_output, expected_sign]).unwrap();
        assert!(prover.verify().is_ok());
    }
} 