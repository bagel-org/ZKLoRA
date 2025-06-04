import unittest
import zklora_halo2
import numpy as np

def flatten_matrix(matrix):
    arr = np.asarray(matrix)
    if arr.size == 0:
        return []
    return arr.flatten().tolist()

def quantize_signed(val, scale=1e4):
    mag = abs(int(round(val * scale)))
    sign = 0 if val >= 0 else 1
    return mag, sign

def flatten_and_quantize(matrix, scale=1e4):
    flat = flatten_matrix(matrix)
    if not flat:
        return [], []
    mags, signs = zip(*(quantize_signed(v, scale) for v in flat))
    return list(mags), list(signs)

class TestZKLoRAHalo2(unittest.TestCase):
    def test_proof_generation_and_verification(self):
        input_data = [1.0, 2.0]
        weight_a = [3.0, 4.0]
        weight_b = [5.0, 6.0]
        scale = 1e4
        input_mags, input_signs = flatten_and_quantize(input_data, scale)
        wa_mags, wa_signs = flatten_and_quantize(weight_a, scale)
        wb_mags, wb_signs = flatten_and_quantize(weight_b, scale)
        # Dummy output for interface; in real use, compute output as in the circuit
        output = [input_data[0] * weight_a[0] * weight_b[0]]
        output_mags, output_signs = flatten_and_quantize(output, scale)
        public_inputs = input_mags + wa_mags + wb_mags + output_mags + input_signs + wa_signs + wb_signs + output_signs
        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertGreater(len(proof), 0)
        result = zklora_halo2.verify_proof(proof, public_inputs)
        self.assertTrue(result)

    def test_empty_inputs(self):
        scale = 1e4
        input_data, weight_a, weight_b, output = [], [], [], []
        input_mags, input_signs = flatten_and_quantize(input_data, scale)
        wa_mags, wa_signs = flatten_and_quantize(weight_a, scale)
        wb_mags, wb_signs = flatten_and_quantize(weight_b, scale)
        output_mags, output_signs = flatten_and_quantize(output, scale)
        public_inputs = input_mags + wa_mags + wb_mags + output_mags + input_signs + wa_signs + wb_signs + output_signs
        proof = zklora_halo2.generate_proof([], [], [])
        self.assertIsInstance(proof, bytes)
        self.assertGreater(len(proof), 0)
        result = zklora_halo2.verify_proof(proof, public_inputs)
        self.assertTrue(result)

    def test_large_inputs(self):
        input_data = [float(i) for i in range(100)]
        weight_a = [float(i) for i in range(100, 200)]
        weight_b = [float(i) for i in range(200, 300)]
        scale = 1e4
        input_mags, input_signs = flatten_and_quantize(input_data, scale)
        wa_mags, wa_signs = flatten_and_quantize(weight_a, scale)
        wb_mags, wb_signs = flatten_and_quantize(weight_b, scale)
        # Dummy output for interface; in real use, compute output as in the circuit
        output = [input_data[0] * weight_a[0] * weight_b[0]]
        output_mags, output_signs = flatten_and_quantize(output, scale)
        public_inputs = input_mags + wa_mags + wb_mags + output_mags + input_signs + wa_signs + wb_signs + output_signs
        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertGreater(len(proof), 0)
        result = zklora_halo2.verify_proof(proof, public_inputs)
        self.assertTrue(result)

    def test_negative_inputs(self):
        input_data = [-1.0, -2.0]
        weight_a = [-3.0, -4.0]
        weight_b = [-5.0, -6.0]
        scale = 1e4
        input_mags, input_signs = flatten_and_quantize(input_data, scale)
        wa_mags, wa_signs = flatten_and_quantize(weight_a, scale)
        wb_mags, wb_signs = flatten_and_quantize(weight_b, scale)
        # Dummy output for interface; in real use, compute output as in the circuit
        output = [input_data[0] * weight_a[0] * weight_b[0]]
        output_mags, output_signs = flatten_and_quantize(output, scale)
        public_inputs = input_mags + wa_mags + wb_mags + output_mags + input_signs + wa_signs + wb_signs + output_signs
        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertGreater(len(proof), 0)
        result = zklora_halo2.verify_proof(proof, public_inputs)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 