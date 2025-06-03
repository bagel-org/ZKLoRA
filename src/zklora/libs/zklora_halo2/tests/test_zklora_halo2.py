import unittest
import zklora_halo2

class TestZKLoRAHalo2(unittest.TestCase):
    def test_proof_generation_and_verification(self):
        input_data = [1.0, 2.0]
        weight_a = [3.0, 4.0]
        weight_b = [5.0, 6.0]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)  # Check dummy proof size

        result = zklora_halo2.verify_proof(proof, [1.0, 2.0])
        self.assertTrue(result)

    def test_empty_inputs(self):
        proof = zklora_halo2.generate_proof([], [], [])
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, [])
        self.assertTrue(result)

    def test_large_inputs(self):
        input_data = [float(i) for i in range(100)]
        weight_a = [float(i) for i in range(100, 200)]
        weight_b = [float(i) for i in range(200, 300)]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, input_data)
        self.assertTrue(result)

    def test_negative_inputs(self):
        input_data = [-1.0, -2.0]
        weight_a = [-3.0, -4.0]
        weight_b = [-5.0, -6.0]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, input_data)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 