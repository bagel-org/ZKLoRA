from __future__ import annotations

import pytest
import json
import tempfile
import os

from zklora.halo2_low_rank_chip import (
    LowRankChip,
    BatchedLookupOptimizer,
    create_optimized_chip_for_model
)


@pytest.mark.unit
class TestLowRankChip:
    
    def test_initialization(self):
        chip = LowRankChip(rank=16, weight_bits=4, activation_bits=8)
        
        assert chip.rank == 16
        assert chip.weight_bits == 4
        assert chip.activation_bits == 8
        
        assert chip.advice_columns['input_activations'] == 16
        assert chip.advice_columns['intermediate'] == 16
        assert chip.advice_columns['lora_weights_a'] == 16
        assert chip.advice_columns['lora_weights_b'] == 16
        assert chip.advice_columns['output'] == 1
        assert chip.advice_columns['base_commitment'] == 1
        
        assert chip.lookup_columns['weight_value'] == 1
        assert chip.lookup_columns['activation_value'] == 1
        assert chip.lookup_columns['product_result'] == 1
    
    def test_generate_chip_config(self):
        chip = LowRankChip(rank=8)
        config = chip.generate_chip_config()
        
        assert config['chip_type'] == 'low_rank_lora'
        assert config['rank'] == 8
        assert 'quantization' in config
        assert config['quantization']['weight_bits'] == 4
        assert config['quantization']['activation_bits'] == 8
        assert 'columns' in config
        assert 'gates' in config
        assert 'lookup_tables' in config
    
    def test_advice_layout(self):
        chip = LowRankChip(rank=4)
        layout = chip._get_advice_layout()
        
        expected_columns = (
            4 + 4 + 4 + 4 + 1 + 1
        )
        assert len(layout) == expected_columns
        
        input_act_columns = [col for col in layout if 'input_act_' in col['name']]
        assert len(input_act_columns) == 4
        
        for col in input_act_columns:
            assert col['type'] == 'activation'
            assert col['bits'] == 8
    
    def test_lookup_layout(self):
        chip = LowRankChip(rank=16)
        layout = chip._get_lookup_layout()
        
        assert len(layout) == 3
        assert layout[0]['name'] == 'lookup_weight'
        assert layout[0]['bits'] == 4
        assert layout[1]['name'] == 'lookup_activation'
        assert layout[1]['bits'] == 8
        assert layout[2]['name'] == 'lookup_result'
        assert layout[2]['bits'] == 16
    
    def test_fixed_layout(self):
        chip = LowRankChip(rank=16)
        layout = chip._get_fixed_layout()
        
        assert len(layout) == 3
        assert all(col['type'] == 'selector' for col in layout)
        assert any(col['name'] == 'selector_lowrank' for col in layout)
        assert any(col['name'] == 'selector_lookup' for col in layout)
        assert any(col['name'] == 'selector_commitment' for col in layout)
    
    def test_custom_gates(self):
        chip = LowRankChip(rank=8)
        gates = chip._define_custom_gates()
        
        assert len(gates) == 3
        
        gate_names = [gate['name'] for gate in gates]
        assert 'low_rank_matvec_1' in gate_names
        assert 'low_rank_matvec_2' in gate_names
        assert 'verify_base_commitment' in gate_names
        
        for gate in gates:
            assert 'type' in gate
            assert 'constraints' in gate
            assert 'selector' in gate
    
    def test_lookup_tables(self):
        chip = LowRankChip(rank=16, weight_bits=4, activation_bits=8)
        tables = chip._define_lookup_tables()
        
        assert 'multiplication_table' in tables
        mult_table = tables['multiplication_table']
        
        assert 'columns' in mult_table
        assert 'entries' in mult_table
        assert len(mult_table['entries']) == 16 * 256
        
        sample_entry = mult_table['entries'][0]
        assert 'weight' in sample_entry
        assert 'activation' in sample_entry
        assert 'result' in sample_entry
    
    def test_export_chip_definition(self):
        chip = LowRankChip(rank=4)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config = chip.export_chip_definition(f.name)
            
        assert os.path.exists(f.name)
        
        with open(f.name, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['chip_type'] == 'low_rank_lora'
        assert loaded_config['rank'] == 4
        
        os.unlink(f.name)


@pytest.mark.unit
class TestBatchedLookupOptimizer:
    
    def test_initialization(self):
        optimizer = BatchedLookupOptimizer(batch_size=16)
        assert optimizer.batch_size == 16
    
    def test_optimize_lookup_layout(self):
        optimizer = BatchedLookupOptimizer(batch_size=8)
        layout = optimizer.optimize_lookup_layout(num_operations=100)
        
        assert 'batched_lookups' in layout
        batched = layout['batched_lookups']
        
        assert batched['rows_per_batch'] == 1
        assert batched['lookups_per_row'] == 8
        assert batched['total_batches'] == 13
        assert batched['columns_per_lookup'] == 3
        assert batched['total_columns'] == 24
    
    def test_optimize_lookup_layout_edge_cases(self):
        optimizer = BatchedLookupOptimizer(batch_size=10)
        
        layout_exact = optimizer.optimize_lookup_layout(num_operations=50)
        assert layout_exact['batched_lookups']['total_batches'] == 5
        
        layout_partial = optimizer.optimize_lookup_layout(num_operations=51)
        assert layout_partial['batched_lookups']['total_batches'] == 6
    
    def test_generate_batched_constraints(self):
        optimizer = BatchedLookupOptimizer(batch_size=4)
        constraints = optimizer.generate_batched_constraints(batch_size=4)
        
        assert len(constraints) == 4
        
        for i, constraint in enumerate(constraints):
            assert constraint['lookup_index'] == i
            assert constraint['weight_column'] == f'batch_weight_{i}'
            assert constraint['activation_column'] == f'batch_activation_{i}'
            assert constraint['result_column'] == f'batch_result_{i}'
            assert constraint['table'] == 'multiplication_table'


@pytest.mark.unit
def test_create_optimized_chip_for_model():
    model_config = {
        'optimizations': {
            'rank': 32,
            'weight_quantization': {
                'weight_bits': 4
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        chip_path = create_optimized_chip_for_model(
            model_config, output_dir=tmpdir
        )
        
        assert os.path.exists(chip_path)
        assert chip_path.endswith('low_rank_chip_r32.json')
        
        with open(chip_path, 'r') as f:
            chip_config = json.load(f)
        
        assert chip_config['rank'] == 32
        assert chip_config['quantization']['weight_bits'] == 4
        assert 'batched_lookups' in chip_config


@pytest.mark.unit
def test_chip_with_different_ranks():
    for rank in [4, 8, 16, 32, 64]:
        chip = LowRankChip(rank=rank)
        config = chip.generate_chip_config()
        
        assert config['rank'] == rank
        
        advice_layout = chip._get_advice_layout()
        input_columns = [
            col for col in advice_layout 
            if col['name'].startswith('input_act_')
        ]
        assert len(input_columns) == rank


@pytest.mark.unit
def test_chip_with_different_quantization():
    test_cases = [
        (2, 4),
        (4, 8),
        (8, 8),
        (4, 16)
    ]
    
    for weight_bits, activation_bits in test_cases:
        chip = LowRankChip(
            rank=8, 
            weight_bits=weight_bits, 
            activation_bits=activation_bits
        )
        config = chip.generate_chip_config()
        
        assert config['quantization']['weight_bits'] == weight_bits
        assert config['quantization']['activation_bits'] == activation_bits
        
        tables = chip._define_lookup_tables()
        expected_entries = (2 ** weight_bits) * (2 ** activation_bits)
        assert len(tables['multiplication_table']['entries']) == expected_entries 