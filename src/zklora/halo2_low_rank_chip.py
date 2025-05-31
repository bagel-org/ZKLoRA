"""
Halo2 chip definition for low-rank LoRA operations.
This chip implements custom gates for efficient low-rank matrix operations
using lookup tables instead of multiplication gates.
"""

from typing import List, Dict, Tuple
import json


class LowRankChip:
    """
    Custom Halo2 chip for low-rank LoRA operations.
    
    Key optimizations:
    1. Dedicated advice columns for rank-r operations
    2. Lookup tables for 4-bit × 8-bit multiplications
    3. Batched lookup operations
    4. External commitment verification for base model
    """
    
    def __init__(self, rank: int, weight_bits: int = 4, activation_bits: int = 8):
        self.rank = rank
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Define advice columns
        self.advice_columns = {
            'input_activations': rank,      # r columns for B^T x
            'intermediate': rank,            # r columns for intermediate result
            'lora_weights_a': rank,          # r columns for A matrix
            'lora_weights_b': rank,          # r columns for B matrix
            'output': 1,                     # Output column
            'base_commitment': 1,            # Base model commitment
        }
        
        # Define lookup columns
        self.lookup_columns = {
            'weight_value': 1,
            'activation_value': 1,
            'product_result': 1,
        }
        
    def generate_chip_config(self) -> Dict:
        """Generate the chip configuration for EZKL"""
        config = {
            'chip_type': 'low_rank_lora',
            'rank': self.rank,
            'quantization': {
                'weight_bits': self.weight_bits,
                'activation_bits': self.activation_bits,
            },
            'columns': {
                'advice': self._get_advice_layout(),
                'lookup': self._get_lookup_layout(),
                'fixed': self._get_fixed_layout(),
            },
            'gates': self._define_custom_gates(),
            'lookup_tables': self._define_lookup_tables(),
        }
        
        return config
    
    def _get_advice_layout(self) -> List[Dict]:
        """Define advice column layout"""
        layout = []
        
        # Input activation columns
        for i in range(self.rank):
            layout.append({
                'name': f'input_act_{i}',
                'type': 'activation',
                'bits': self.activation_bits,
            })
        
        # Intermediate result columns
        for i in range(self.rank):
            layout.append({
                'name': f'intermediate_{i}',
                'type': 'intermediate',
                'bits': 16,  # Higher precision for intermediate
            })
        
        # Weight columns for A and B
        for matrix in ['A', 'B']:
            for i in range(self.rank):
                layout.append({
                    'name': f'weight_{matrix}_{i}',
                    'type': 'weight',
                    'bits': self.weight_bits,
                })
        
        # Output and commitment columns
        layout.extend([
            {'name': 'output', 'type': 'output', 'bits': 32},
            {'name': 'base_commitment', 'type': 'commitment', 'bits': 256},
        ])
        
        return layout
    
    def _get_lookup_layout(self) -> List[Dict]:
        """Define lookup column layout"""
        return [
            {'name': 'lookup_weight', 'bits': self.weight_bits},
            {'name': 'lookup_activation', 'bits': self.activation_bits},
            {'name': 'lookup_result', 'bits': 16},
        ]
    
    def _get_fixed_layout(self) -> List[Dict]:
        """Define fixed column layout"""
        return [
            {'name': 'selector_lowrank', 'type': 'selector'},
            {'name': 'selector_lookup', 'type': 'selector'},
            {'name': 'selector_commitment', 'type': 'selector'},
        ]
    
    def _define_custom_gates(self) -> List[Dict]:
        """Define custom gates for low-rank operations"""
        gates = []
        
        # Gate 1: Low-rank matrix-vector product using lookups
        # Computes z = B^T x using lookup tables
        gates.append({
            'name': 'low_rank_matvec_1',
            'type': 'custom',
            'constraints': [
                # For each element of z[i] = sum_j B[j,i] * x[j]
                # We use lookup tables instead of multiplication
                {
                    'lookup': 'multiplication_table',
                    'inputs': ['weight_B_{i}', 'input_act_{j}'],
                    'output': 'intermediate_{i}',
                }
            ],
            'selector': 'selector_lowrank',
        })
        
        # Gate 2: Second matrix-vector product
        # Computes y = A z using lookup tables
        gates.append({
            'name': 'low_rank_matvec_2',
            'type': 'custom',
            'constraints': [
                {
                    'lookup': 'multiplication_table',
                    'inputs': ['weight_A_{i}', 'intermediate_{j}'],
                    'output': 'output',
                }
            ],
            'selector': 'selector_lowrank',
        })
        
        # Gate 3: Base model commitment verification
        gates.append({
            'name': 'verify_base_commitment',
            'type': 'custom',
            'constraints': [
                {
                    'type': 'poseidon_hash',
                    'input': 'base_activations',
                    'expected': 'base_commitment',
                }
            ],
            'selector': 'selector_commitment',
        })
        
        return gates
    
    def _define_lookup_tables(self) -> Dict:
        """Define lookup tables for multiplication"""
        # Create multiplication lookup table
        table_entries = []
        
        # Generate all possible weight × activation combinations
        weight_range = range(-(2**(self.weight_bits-1)), 2**(self.weight_bits-1))
        activation_range = range(0, 2**self.activation_bits)
        
        for w in weight_range:
            for a in activation_range:
                table_entries.append({
                    'weight': w,
                    'activation': a,
                    'result': w * a,
                })
        
        return {
            'multiplication_table': {
                'columns': ['lookup_weight', 'lookup_activation', 'lookup_result'],
                'entries': table_entries,
            }
        }
    
    def export_chip_definition(self, output_path: str):
        """Export chip definition for use in proof generation"""
        config = self.generate_chip_config()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Exported Halo2 chip definition to {output_path}")
        return config


class BatchedLookupOptimizer:
    """
    Optimizes lookup operations by batching multiple lookups per row.
    This reduces the total number of rows needed in the circuit.
    """
    
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
    
    def optimize_lookup_layout(self, num_operations: int) -> Dict:
        """
        Create optimized layout for batched lookups.
        Instead of one lookup per row, pack multiple lookups.
        """
        num_batches = (num_operations + self.batch_size - 1) // self.batch_size
        
        layout = {
            'batched_lookups': {
                'rows_per_batch': 1,
                'lookups_per_row': self.batch_size,
                'total_batches': num_batches,
                'columns_per_lookup': 3,  # weight, activation, result
                'total_columns': 3 * self.batch_size,
            }
        }
        
        return layout
    
    def generate_batched_constraints(self, batch_size: int) -> List[Dict]:
        """Generate constraints for batched lookup operations"""
        constraints = []
        
        for i in range(batch_size):
            constraints.append({
                'lookup_index': i,
                'weight_column': f'batch_weight_{i}',
                'activation_column': f'batch_activation_{i}',
                'result_column': f'batch_result_{i}',
                'table': 'multiplication_table',
            })
        
        return constraints


def create_optimized_chip_for_model(
    model_config: Dict,
    output_dir: str = "chip_configs"
) -> str:
    """
    Create an optimized Halo2 chip configuration for a specific model.
    
    Args:
        model_config: Configuration from the optimized LoRA export
        output_dir: Directory to save chip configuration
        
    Returns:
        Path to the exported chip configuration
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters from model config
    rank = model_config['optimizations']['rank']
    weight_bits = model_config['optimizations']['weight_quantization']['weight_bits']
    
    # Create chip
    chip = LowRankChip(rank=rank, weight_bits=weight_bits)
    
    # Add batched lookup optimization
    batch_optimizer = BatchedLookupOptimizer(batch_size=8)
    chip_config = chip.generate_chip_config()
    
    # Add batched lookup configuration
    num_ops = rank * 2  # Two matrix-vector products
    chip_config['batched_lookups'] = batch_optimizer.optimize_lookup_layout(num_ops)
    
    # Export
    output_path = os.path.join(output_dir, f"low_rank_chip_r{rank}.json")
    with open(output_path, 'w') as f:
        json.dump(chip_config, f, indent=2)
    
    return output_path 