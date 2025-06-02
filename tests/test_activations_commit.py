from __future__ import annotations

import pytest
from unittest.mock import patch, Mock

from zklora.activations_commit import commit_activations, verify_commitment


@pytest.mark.unit
class TestActivationsCommit:
    
    @patch('zklora.activations_commit.activations_commitment')
    def test_commit_activations(self, mock_commit):
        mock_commit.return_value = "test_commitment_hash"
        
        result = commit_activations("test_file.json")
        
        assert result == "test_commitment_hash"
        mock_commit.assert_called_once_with("test_file.json")
    
    @patch('zklora.activations_commit.activations_verify_commitment')
    def test_verify_commitment(self, mock_verify):
        mock_verify.return_value = True
        
        result = verify_commitment("test_file.json", "test_commitment")
        
        assert result is True
        mock_verify.assert_called_once_with("test_file.json", "test_commitment") 