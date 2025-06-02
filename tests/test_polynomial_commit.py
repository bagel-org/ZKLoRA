from __future__ import annotations

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock

from zklora.polynomial_commit import (
    _hash_leaf,
    _parent_hash, 
    _merkle_root,
    commit_activations,
    verify_commitment,
    LEAF_EMPTY
)


@pytest.mark.unit
class TestHashLeaf:
    
    def test_hash_leaf_int(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _hash_leaf(42, nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32  # BLAKE3 output size
    
    def test_hash_leaf_float(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _hash_leaf(3.14, nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_hash_leaf_deterministic(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        
        result1 = _hash_leaf(42, nonce)
        result2 = _hash_leaf(42, nonce)
        
        assert result1 == result2
    
    def test_hash_leaf_different_nonce(self):
        nonce1 = b"test_nonce_32_bytes_padded_here!"
        nonce2 = b"different_nonce_32_bytes_padded!"
        
        result1 = _hash_leaf(42, nonce1)
        result2 = _hash_leaf(42, nonce2)
        
        assert result1 != result2


@pytest.mark.unit
class TestParentHash:
    
    def test_parent_hash_basic(self):
        left = b"left_hash_32_bytes_padded_here!!"
        right = b"right_hash_32_bytes_padded_here!"
        
        result = _parent_hash(left, right)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_parent_hash_deterministic(self):
        left = b"left_hash_32_bytes_padded_here!!"
        right = b"right_hash_32_bytes_padded_here!"
        
        result1 = _parent_hash(left, right)
        result2 = _parent_hash(left, right)
        
        assert result1 == result2
    
    def test_parent_hash_order_matters(self):
        left = b"left_hash_32_bytes_padded_here!!"
        right = b"right_hash_32_bytes_padded_here!"
        
        result1 = _parent_hash(left, right)
        result2 = _parent_hash(right, left)
        
        assert result1 != result2


@pytest.mark.unit
class TestMerkleRoot:
    
    def test_merkle_root_empty(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _merkle_root([], nonce)
        
        assert result == LEAF_EMPTY
    
    def test_merkle_root_single_value(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _merkle_root([42], nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
        assert result != LEAF_EMPTY
    
    def test_merkle_root_two_values(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _merkle_root([1, 2], nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_merkle_root_odd_length(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        result = _merkle_root([1, 2, 3], nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32
    
    def test_merkle_root_deterministic(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        values = [1, 2, 3, 4]
        
        result1 = _merkle_root(values, nonce)
        result2 = _merkle_root(values, nonce)
        
        assert result1 == result2
    
    def test_merkle_root_mixed_types(self):
        nonce = b"test_nonce_32_bytes_padded_here!"
        values = [1, 2.5, 3, 4.7]
        
        result = _merkle_root(values, nonce)
        
        assert isinstance(result, bytes)
        assert len(result) == 32


@pytest.mark.unit
class TestCommitActivations:
    
    def test_commit_activations_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3, 4]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = commit_activations(temp_path)
            
            assert isinstance(result, str)
            commitment_data = json.loads(result)
            assert "root" in commitment_data
            assert "nonce" in commitment_data
            assert commitment_data["root"].startswith("0x")
            assert commitment_data["nonce"].startswith("0x")
            assert len(commitment_data["root"]) == 66  # 0x + 64 hex chars
            assert len(commitment_data["nonce"]) == 66  # 0x + 64 hex chars
        finally:
            os.unlink(temp_path)
    
    def test_commit_activations_nested_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [[1, 2], [3, [4, 5]], 6]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = commit_activations(temp_path)
            
            commitment_data = json.loads(result)
            assert "root" in commitment_data
            assert "nonce" in commitment_data
        finally:
            os.unlink(temp_path)
    
    @patch('numpy.asarray')
    def test_commit_activations_numpy_fallback(self, mock_asarray):
        mock_asarray.side_effect = Exception("Numpy not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = commit_activations(temp_path)
            
            commitment_data = json.loads(result)
            assert "root" in commitment_data
            assert "nonce" in commitment_data
        finally:
            os.unlink(temp_path)
    
    def test_commit_activations_empty_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": []}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = commit_activations(temp_path)
            
            commitment_data = json.loads(result)
            assert "root" in commitment_data
            assert "nonce" in commitment_data
        finally:
            os.unlink(temp_path)

    @patch('numpy.asarray')
    def test_commit_activations_numpy_fallback_with_exception(self, mock_asarray):
        # Test the fallback when numpy import itself fails
        mock_asarray.side_effect = ImportError("No module named 'numpy'")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = commit_activations(temp_path)
            
            commitment_data = json.loads(result)
            assert "root" in commitment_data
            assert "nonce" in commitment_data
        finally:
            os.unlink(temp_path)


@pytest.mark.unit 
class TestVerifyCommitment:
    
    def test_verify_commitment_valid(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3, 4]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            # First commit
            commitment = commit_activations(temp_path)
            
            # Then verify
            result = verify_commitment(temp_path, commitment)
            
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_verify_commitment_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = verify_commitment(temp_path, "invalid json")
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_verify_commitment_missing_fields(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            invalid_commitment = json.dumps({"root": "0x123"})  # missing nonce
            result = verify_commitment(temp_path, invalid_commitment)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_verify_commitment_invalid_hex(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            invalid_commitment = json.dumps({
                "root": "0xzzzz",  # invalid hex
                "nonce": "0x1234567890abcdef" + "0" * 48
            })
            result = verify_commitment(temp_path, invalid_commitment)
            
            assert result is False
        finally:
            os.unlink(temp_path)
    
    def test_verify_commitment_wrong_data(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f1:
            data1 = {"input_data": [1, 2, 3]}
            json.dump(data1, f1)
            temp_path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f2:
            data2 = {"input_data": [4, 5, 6]}  # different data
            json.dump(data2, f2)
            temp_path2 = f2.name
        
        try:
            # Commit to first file
            commitment = commit_activations(temp_path1)
            
            # Try to verify against second file
            result = verify_commitment(temp_path2, commitment)
            
            assert result is False
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)
    
    def test_verify_commitment_hex_case_insensitive(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            commitment = commit_activations(temp_path)
            commitment_data = json.loads(commitment)
            
            # Convert to uppercase
            commitment_data_upper = {
                "root": commitment_data["root"].upper(),
                "nonce": commitment_data["nonce"].upper()
            }
            commitment_upper = json.dumps(commitment_data_upper)
            
            result = verify_commitment(temp_path, commitment_upper)
            
            assert result is True
        finally:
            os.unlink(temp_path)
    
    @patch('numpy.asarray')
    def test_verify_commitment_numpy_fallback(self, mock_asarray):
        mock_asarray.side_effect = Exception("Numpy not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            # Commit with numpy fallback
            commitment = commit_activations(temp_path)
            
            # Verify with numpy fallback
            result = verify_commitment(temp_path, commitment)
            
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_verify_commitment_no_hex_prefix(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"input_data": [1, 2, 3]}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            commitment = commit_activations(temp_path)
            commitment_data = json.loads(commitment)
            
            # Remove 0x prefix manually
            commitment_data_no_prefix = {
                "root": commitment_data["root"][2:],  # Remove 0x
                "nonce": commitment_data["nonce"][2:]  # Remove 0x
            }
            commitment_no_prefix = json.dumps(commitment_data_no_prefix)
            
            result = verify_commitment(temp_path, commitment_no_prefix)
            
            assert result is True
        finally:
            os.unlink(temp_path) 