use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector},
    poly::Rotation,
};

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

impl Circuit<halo2_proofs::pasta::Fp> for LoRACircuit {
    type Config = LoRAConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<halo2_proofs::pasta::Fp>) -> Self::Config {
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
        mut layouter: impl Layouter<halo2_proofs::pasta::Fp>,
    ) -> Result<(), Error> {
        let input_val = if !self.input.is_empty() {
            halo2_proofs::pasta::Fp::from(self.input[0].abs() as u64)
        } else {
            halo2_proofs::pasta::Fp::zero()
        };

        let weight_a_val = if !self.weight_a.is_empty() {
            halo2_proofs::pasta::Fp::from(self.weight_a[0].abs() as u64)
        } else {
            halo2_proofs::pasta::Fp::one()
        };

        let weight_b_val = if !self.weight_b.is_empty() {
            halo2_proofs::pasta::Fp::from(self.weight_b[0].abs() as u64)
        } else {
            halo2_proofs::pasta::Fp::one()
        };

        let output_val = input_val * weight_a_val * weight_b_val;

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

                // region.assign_advice_from_constant(
                //     || "output",
                //     config.output,
                //     0,
                //     output_val,
                // )?;

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

        let expected_output = vec![halo2_proofs::pasta::Fp::from(6u64)];
        let prover = MockProver::run(4, &circuit, vec![expected_output]).unwrap();
        assert!(prover.verify().is_ok());

        let wrong_output = vec![halo2_proofs::pasta::Fp::from(7u64)];
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

        let expected_output = vec![halo2_proofs::pasta::Fp::zero()];
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

        let expected_output = vec![halo2_proofs::pasta::Fp::from(6u64)];
        let prover = MockProver::run(4, &circuit, vec![expected_output]).unwrap();
        assert!(prover.verify().is_ok());
    }
} 