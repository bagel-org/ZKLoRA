use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value, AssignedCell},
    pasta::Fp,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};
use ff::Field;

use crate::{dequantize_from_field, quantize_to_field, Quantized, SCALE_FACTOR};

/// Circuit configuration shared by every row in the region.
#[derive(Clone)]
pub struct LoRAConfig {
    lhs_mag: Column<Advice>,
    lhs_sign: Column<Advice>,
    rhs_mag: Column<Advice>,
    rhs_sign: Column<Advice>,
    prod_mag: Column<Advice>,
    prod_sign: Column<Advice>,
    partial: Column<Advice>,
    output_mag: Column<Instance>,
    output_sign: Column<Instance>,
    sel_mul: Selector,
    sel_acc: Selector,
    sel_out: Selector,
}

/// Flattened LoRA layer (input vector `x`, rank-r matrix `A`, matrix `B`).
/// Shapes are derived from the slice lengths, so callers do **not** need to
/// supply explicit dimensions.  All numbers are fixed-point with `SCALE_FACTOR`.
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
        // Advice columns hold the witnesses for one multiplication + prefix sum.
        let lhs_mag = meta.advice_column();
        let lhs_sign = meta.advice_column();
        let rhs_mag = meta.advice_column();
        let rhs_sign = meta.advice_column();
        let prod_mag = meta.advice_column();
        let prod_sign = meta.advice_column();
        let partial = meta.advice_column();

        // Public instance columns – final output magnitude & sign per row.
        let output_mag = meta.instance_column();
        let output_sign = meta.instance_column();

        // Every column participates in equality constraints at least once.
        for col in [
            lhs_mag,
            lhs_sign,
            rhs_mag,
            rhs_sign,
            prod_mag,
            prod_sign,
            partial,
        ] {
            meta.enable_equality(col);
        }
        meta.enable_equality(output_mag);
        meta.enable_equality(output_sign);

        // Selectors
        let sel_mul = meta.selector();
        let sel_acc = meta.selector();
        let sel_out = meta.selector();

        // Gate 1: fixed-point multiplication + XOR sign.
        meta.create_gate("mul_gate", |meta| {
            let s = meta.query_selector(sel_mul);

            let lhs_mag_e = meta.query_advice(lhs_mag, Rotation::cur());
            let rhs_mag_e = meta.query_advice(rhs_mag, Rotation::cur());
            let prod_mag_e = meta.query_advice(prod_mag, Rotation::cur());

            let lhs_sign_e = meta.query_advice(lhs_sign, Rotation::cur());
            let rhs_sign_e = meta.query_advice(rhs_sign, Rotation::cur());
            let prod_sign_e = meta.query_advice(prod_sign, Rotation::cur());

            let scale = Expression::Constant(Fp::from(SCALE_FACTOR));
            let two = Expression::Constant(Fp::from(2));

            // Magnitude: lhs * rhs = product * SCALE_FACTOR
            let mag_constraint = lhs_mag_e * rhs_mag_e - prod_mag_e.clone() * scale;
            // Sign   : XOR encoded in the field (lhs ⊕ rhs = prod_sign)
            let xor_expr = lhs_sign_e.clone() + rhs_sign_e.clone()
                - two.clone() * lhs_sign_e.clone() * rhs_sign_e.clone()
                - prod_sign_e;
            let sign_constraint = xor_expr * prod_mag_e.clone();

            vec![s.clone() * mag_constraint, s * sign_constraint]
        });

        // Gate 2: running prefix sum of signed products.
        meta.create_gate("acc_gate", |meta| {
            let s = meta.query_selector(sel_acc);

            let partial_cur = meta.query_advice(partial, Rotation::cur());
            let partial_prev = meta.query_advice(partial, Rotation::prev());
            let prod_mag_e = meta.query_advice(prod_mag, Rotation::cur());
            let prod_sign_e = meta.query_advice(prod_sign, Rotation::cur());

            let two = Expression::Constant(Fp::from(2));
            // signed_prod = (1 − 2·sign) · prod_mag
            let signed_prod = prod_mag_e.clone() * (Expression::Constant(Fp::ONE) - two * prod_sign_e);

            let acc_constraint = partial_cur - partial_prev - signed_prod;
            vec![s * acc_constraint]
        });

        // Gate 3: output_gate
        meta.create_gate("output_gate", |meta| {
            let s = meta.query_selector(sel_out);
            let mag = meta.query_advice(prod_mag, Rotation::cur());
            let sign = meta.query_advice(prod_sign, Rotation::cur());
            let partial_cur = meta.query_advice(partial, Rotation::cur());
            let two = Expression::Constant(Fp::from(2));
            let expr = (Expression::Constant(Fp::ONE) - two * sign) * mag - partial_cur;
            vec![s * expr]
        });

        LoRAConfig {
            lhs_mag,
            lhs_sign,
            rhs_mag,
            rhs_sign,
            prod_mag,
            prod_sign,
            partial,
            output_mag,
            output_sign,
            sel_mul,
            sel_acc,
            sel_out,
        }
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<Fp>) -> Result<(), Error> {
        // Shapes inferred from slice lengths.
        if self.input.is_empty() {
            return Err(Error::Synthesis);
        }
        let cols = self.input.len();
        if self.weight_a.len() % cols != 0 {
            return Err(Error::Synthesis);
        }
        let rank = self.weight_a.len() / cols;
        if self.weight_b.len() % rank != 0 {
            return Err(Error::Synthesis);
        }
        let rows = self.weight_b.len() / rank;

        // Quantise all inputs in advance so we can reuse them.
        let q_input: Vec<Quantized> = self.input.iter().copied().map(quantize_to_field).collect();
        let q_a: Vec<Quantized> = self.weight_a.iter().copied().map(quantize_to_field).collect();
        let q_b: Vec<Quantized> = self.weight_b.iter().copied().map(quantize_to_field).collect();

        // Helper to obtain signed Fp from (mag, sign).
        let to_signed = |q: &Quantized| {
            if q.sign == Fp::ONE {
                Fp::ZERO - q.magnitude
            } else {
                q.magnitude
            }
        };

        // Precompute v = A·x in the real domain (no scaling division).
        let mut v_floats = vec![0.0f64; rank];
        for j in 0..rank {
            let mut acc = 0.0;
            for k in 0..cols {
                acc += self.weight_a[j * cols + k] * self.input[k];
            }
            v_floats[j] = acc;
        }
        let v_quant: Vec<Quantized> = v_floats.iter().copied().map(quantize_to_field).collect();

        // Precompute y = B·v.
        let mut y_floats = vec![0.0f64; rows];
        for i in 0..rows {
            let mut acc = 0.0;
            for j in 0..rank {
                acc += self.weight_b[i * rank + j] * v_floats[j];
            }
            y_floats[i] = acc;
        }
        let y_quant: Vec<Quantized> = y_floats.iter().copied().map(quantize_to_field).collect();

        // Storage for public output cells to constrain after the region is laid out.
        let mut public_cells: Vec<(usize, AssignedCell<Fp, Fp>, AssignedCell<Fp, Fp>)> = Vec::new();

        // Lay out the entire computation in one region for simplicity.
        layouter.assign_region(|| "lora_matrix_mul", |mut region| {
            let mut offset: usize = 0;

            // 1) Compute v = A·x (rank dot-products).
            for j in 0..rank {
                let mut partial_val = Fp::ZERO;
                for k in 0..cols {
                    let a_q = &q_a[j * cols + k];
                    let x_q = &q_input[k];

                    // product = a * x  (already in real domain)
                    let product_f = self.weight_a[j * cols + k] * self.input[k];
                    let product_q = quantize_to_field(product_f);

                    // Assign witnesses.
                    region.assign_advice(|| "lhs_mag", config.lhs_mag, offset, || Value::known(a_q.magnitude))?;
                    region.assign_advice(|| "lhs_sign", config.lhs_sign, offset, || Value::known(a_q.sign))?;
                    region.assign_advice(|| "rhs_mag", config.rhs_mag, offset, || Value::known(x_q.magnitude))?;
                    region.assign_advice(|| "rhs_sign", config.rhs_sign, offset, || Value::known(x_q.sign))?;
                    region.assign_advice(|| "prod_mag", config.prod_mag, offset, || Value::known(product_q.magnitude))?;
                    region.assign_advice(|| "prod_sign", config.prod_sign, offset, || Value::known(product_q.sign))?;

                    // Update running sum and write to `partial` column.
                    let signed_prod = to_signed(&product_q);
                    partial_val = if k == 0 { signed_prod } else { partial_val + signed_prod };
                    region.assign_advice(|| "partial", config.partial, offset, || Value::known(partial_val))?;

                    // Enable gates
                    config.sel_mul.enable(&mut region, offset)?;
                    if k > 0 {
                        config.sel_acc.enable(&mut region, offset)?;
                    }
                    offset += 1;
                }

                // Dot-product result already precomputed in v_quant; nothing to push here.
            }

            // 2) Compute y = B·v (rows dot-products) and expose as public output.
            for i in 0..rows {
                let mut partial_val = Fp::ZERO;
                for j in 0..rank {
                    let b_q = &q_b[i * rank + j];
                    let v_q = &v_quant[j];

                    // product = b * v  (already in real domain)
                    let product_f = self.weight_b[i * rank + j] * v_floats[j];
                    let product_q = quantize_to_field(product_f);

                    region.assign_advice(|| "lhs_mag", config.lhs_mag, offset, || Value::known(b_q.magnitude))?;
                    region.assign_advice(|| "lhs_sign", config.lhs_sign, offset, || Value::known(b_q.sign))?;
                    region.assign_advice(|| "rhs_mag", config.rhs_mag, offset, || Value::known(v_q.magnitude))?;
                    region.assign_advice(|| "rhs_sign", config.rhs_sign, offset, || Value::known(v_q.sign))?;
                    region.assign_advice(|| "prod_mag", config.prod_mag, offset, || Value::known(product_q.magnitude))?;
                    region.assign_advice(|| "prod_sign", config.prod_sign, offset, || Value::known(product_q.sign))?;

                    let signed_prod = to_signed(&product_q);
                    partial_val = if j == 0 { signed_prod } else { partial_val + signed_prod };
                    region.assign_advice(|| "partial", config.partial, offset, || Value::known(partial_val))?;

                    config.sel_mul.enable(&mut region, offset)?;
                    if j > 0 {
                        config.sel_acc.enable(&mut region, offset)?;
                    }
                    offset += 1;
                }

                // Final output for this row.
                let y_q = y_quant[i];

                // Magnitude witness (no gate on this row).
                let mag_cell = region.assign_advice(|| "output_mag_witness", config.prod_mag, offset, || Value::known(y_q.magnitude))?;
                let sign_cell = region.assign_advice(|| "output_sign_witness", config.prod_sign, offset, || Value::known(y_q.sign))?;
                region.assign_advice(|| "partial_pad", config.partial, offset, || Value::known(partial_val))?;

                // Enable output gate to bind partial, magnitude & sign.
                config.sel_out.enable(&mut region, offset)?;

                // Record cells for public constraints after region.
                public_cells.push((i, mag_cell, sign_cell));

                offset += 1; // move past the output row
            }
            Ok(())
        })?;

        // Constrain the recorded output cells to the instance columns.
        for (row, mag_cell, sign_cell) in public_cells.into_iter() {
            layouter.constrain_instance(mag_cell.cell(), config.output_mag, row)?;
            layouter.constrain_instance(sign_cell.cell(), config.output_sign, row)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::pasta::Fp;

    fn run_prover(circuit: LoRACircuit, expected: Vec<f64>) {
        let expected_q: Vec<Quantized> = expected.into_iter().map(quantize_to_field).collect();
        let mags: Vec<Fp> = expected_q.iter().map(|q| q.magnitude).collect();
        let signs: Vec<Fp> = expected_q.iter().map(|q| q.sign).collect();
        let k = 6; // depth parameter
        let prover = MockProver::run(k, &circuit, vec![mags, signs]).unwrap();
        if let Err(err) = prover.verify() {
            panic!("MockProver failed: {:?}", err);
        }
    }

    #[test]
    fn test_one_by_one() {
        // 1×1 (scalar) should still work for backwards compatibility.
        let circuit = LoRACircuit {
            input: vec![3.0],
            weight_a: vec![4.0], // rank 1 × cols 1
            weight_b: vec![2.0], // rows 1 × rank 1
        };
        let expected = vec![2.0 * 4.0 * 3.0];
        run_prover(circuit, expected);
    }

    #[test]
    fn test_rank1_cols2_rows2() {
        // cols = 2, rank = 1, rows = 2
        let input = vec![1.0, 2.0]; // x ∈ ℝ^{2}
        // A is 1×2, flattened row-major
        let weight_a = vec![0.5, -1.0];
        // B is 2×1, flattened row-major
        let weight_b = vec![1.0, -2.0];

        let circuit = LoRACircuit {
            input: input.clone(),
            weight_a: weight_a.clone(),
            weight_b: weight_b.clone(),
        };

        let v = weight_a[0] * input[0] + weight_a[1] * input[1];
        let y0 = weight_b[0] * v;
        let y1 = weight_b[1] * v;
        run_prover(circuit, vec![y0, y1]);
    }

    #[test]
    fn test_rank2_cols2_rows1() {
        // cols = 2, rank = 2, rows = 1 (full 2×2)
        let input = vec![1.0, -1.0];
        // A : 2×2 (rank 2) -> flattened
        let weight_a = vec![1.0, 0.0, 0.0, 1.0];
        // B : 1×2
        let weight_b = vec![1.0, 1.0];
        let circuit = LoRACircuit {
            input: input.clone(),
            weight_a: weight_a.clone(),
            weight_b: weight_b.clone(),
        };
        let v0 = 1.0 * 1.0 + 0.0 * -1.0;
        let v1 = 0.0 * 1.0 + 1.0 * -1.0;
        let y = weight_b[0] * v0 + weight_b[1] * v1;
        run_prover(circuit, vec![y]);
    }
} 