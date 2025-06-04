import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import sys
from pathlib import Path as _P

# Direct import of activations_commit module
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src" / "zklora"))
import activations_commit # Changed from: from zklora.activations_commit import get_merkle_root
get_merkle_root = activations_commit.get_merkle_root

class TestActivationsCommit(unittest.TestCase):
    @patch('activations_commit.MerkleTree') # Patch target needs to change to local module
    @patch('builtins.open', new_callable=mock_open)
    def test_get_merkle_root_success(self, mock_file_open, MockMerkleTree):
        # Prepare mock data and return values
        activations_data = {"input_data": [[1, 2, 3], [4, 5, 6]]}
        json_data = json.dumps(activations_data)
        mock_file_open.return_value.read.return_value = json_data
        
        expected_merkle_root_hex = "abcdef123456" # Raw hex from merkly
        expected_output = "0x" + expected_merkle_root_hex

        # Configure the mock for MerkleTree
        mock_tree_instance = MagicMock()
        mock_tree_instance.root.hex.return_value = expected_merkle_root_hex
        MockMerkleTree.return_value = mock_tree_instance

        # Call the function
        result = get_merkle_root("dummy_path.json")

        # Assertions
        mock_file_open.assert_called_once_with("dummy_path.json", 'r')
        # Check if MerkleTree was called with the correct (stringified) data
        MockMerkleTree.assert_called_once_with(['1', '2', '3', '4', '5', '6'])
        self.assertEqual(result, expected_output)

    @patch('activations_commit.MerkleTree')
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_get_merkle_root_file_not_found(self, mock_file_open, MockMerkleTree):
        with self.assertRaises(FileNotFoundError):
            get_merkle_root("non_existent_path.json")
        MockMerkleTree.assert_not_called()

    @patch('activations_commit.MerkleTree')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_merkle_root_missing_input_data_key(self, mock_file_open, MockMerkleTree):
        activations_data = {"other_key": [[1, 2, 3]]} # Missing 'input_data'
        json_data = json.dumps(activations_data)
        mock_file_open.return_value.read.return_value = json_data

        with self.assertRaises(KeyError):
            get_merkle_root("dummy_path.json")
        MockMerkleTree.assert_not_called()

    @patch('activations_commit.MerkleTree')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_merkle_root_malformed_json(self, mock_file_open, MockMerkleTree):
        malformed_json_data = "{\"input_data\": [[1, 2, 3], [4, 5, 6]" # Intentionally malformed
        mock_file_open.return_value.read.return_value = malformed_json_data

        with self.assertRaises(json.JSONDecodeError):
            get_merkle_root("dummy_path.json")
        MockMerkleTree.assert_not_called()

    @patch('activations_commit.MerkleTree')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_merkle_root_already_flat_data(self, mock_file_open, MockMerkleTree):
        activations_data = {"input_data": [1, 2, 3, 4, 5, 6]}
        json_data = json.dumps(activations_data)
        mock_file_open.return_value.read.return_value = json_data
        
        expected_merkle_root_hex = "123456abcdef"
        expected_output = "0x" + expected_merkle_root_hex

        mock_tree_instance = MagicMock()
        mock_tree_instance.root.hex.return_value = expected_merkle_root_hex
        MockMerkleTree.return_value = mock_tree_instance

        result = get_merkle_root("dummy_path.json")

        mock_file_open.assert_called_once_with("dummy_path.json", 'r')
        MockMerkleTree.assert_called_once_with(['1', '2', '3', '4', '5', '6'])
        self.assertEqual(result, expected_output)

    # Test for the new empty data handling in get_merkle_root
    @patch('activations_commit.MerkleTree')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_merkle_root_empty_input_data(self, mock_file_open, MockMerkleTree):
        activations_data = {"input_data": []}
        json_data = json.dumps(activations_data)
        mock_file_open.return_value.read.return_value = json_data
        
        # The function now returns a placeholder for empty data *before* calling MerkleTree
        expected_output = "0x" + "0"*64 

        result = get_merkle_root("dummy_path.json")

        mock_file_open.assert_called_once_with("dummy_path.json", 'r')
        # MerkleTree should NOT be called if input_data results in an empty list for the tree
        MockMerkleTree.assert_not_called() 
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main() 