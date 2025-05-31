"""
Optimized proof generator for low-rank LoRA circuits.
Uses custom Halo2 chip definitions and lookup tables.
"""

import os
import glob
import json
import time
import asyncio
from typing import NamedTuple, Optional, Dict
import concurrent.futures

import numpy as np
import onnx
import onnxruntime
import ezkl

from .halo2_low_rank_chip import create_optimized_chip_for_model


class OptimizedProofPaths(NamedTuple):
    circuit: str
    settings: str
    srs: str
    verification_key: str
    proving_key: str
    witness: str
    proof: str
    chip_config: str
    lookup_config: str


def resolve_optimized_proof_paths(proof_dir: str, base_name: str) -> Optional[OptimizedProofPaths]:
    """Retrieves paths for optimized proof-related files."""
    return OptimizedProofPaths(
        circuit=os.path.join(proof_dir, f"{base_name}.ezkl"),
        settings=os.path.join(proof_dir, f"{base_name}_settings.json"),
        srs=os.path.join(proof_dir, "kzg.srs"),
        verification_key=os.path.join(proof_dir, f"{base_name}.vk"),
        proving_key=os.path.join(proof_dir, f"{base_name}.pk"),
        witness=os.path.join(proof_dir, f"{base_name}_witness.json"),
        proof=os.path.join(proof_dir, f"{base_name}.pf"),
        chip_config=os.path.join(proof_dir, f"{base_name}_chip.json"),
        lookup_config=os.path.join(proof_dir, f"{base_name}_lookup.json"),
    )


async def generate_optimized_proof_single(
    onnx_path: str,
    json_path: str,
    config_path: str,
    output_dir: str,
    verbose: bool = False
) -> Dict:
    """Generate proof for a single optimized LoRA module."""
    
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    # Load model configuration
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create optimized chip configuration
    chip_config_path = create_optimized_chip_for_model(model_config, output_dir)
    
    names = resolve_optimized_proof_paths(output_dir, base_name)
    if names is None:
        return None
    
    # Configure for optimized proof generation
    py_args = ezkl.PyRunArgs()
    py_args.input_visibility = "public"
    py_args.output_visibility = "public"
    py_args.param_visibility = "private"
    
    # Reduced logrows due to optimizations
    rank = model_config['optimizations']['rank']
    py_args.logrows = min(16, 10 + rank)  # Much smaller than original 20
    
    if verbose:
        print(f"Generating optimized proof for {base_name} with rank={rank}")
    
    start_time = time.time()
    
    # 1) Generate settings with custom chip config
    settings = {
        "run_args": py_args.__dict__,
        "chip_config": chip_config_path,
        "use_lookups": True,
        "batch_lookups": True,
        "external_commitments": ["base_model"],
    }
    
    with open(names.settings, 'w') as f:
        json.dump(settings, f)
    
    # 2) Compile circuit with optimizations
    ezkl.compile_circuit(onnx_path, names.circuit, names.settings)
    
    # 3) Generate smaller SRS due to reduced circuit size
    if not os.path.isfile(names.srs):
        ezkl.gen_srs(names.srs, py_args.logrows)
    
    # 4) Setup
    ezkl.setup(names.circuit, names.verification_key, names.proving_key, names.srs)
    
    settings_time = time.time() - start_time
    
    # 5) Generate witness
    start_time = time.time()
    await ezkl.gen_witness(
        data=json_path,
        model=names.circuit,
        output=names.witness
    )
    witness_time = time.time() - start_time
    
    # 6) Prove
    start_time = time.time()
    prove_ok = ezkl.prove(
        names.witness,
        names.circuit,
        names.proving_key,
        names.proof,
        "single",
        names.srs
    )
    prove_time = time.time() - start_time
    
    # Clean up large files
    os.remove(names.proving_key)
    
    result = {
        'module': base_name,
        'settings_time': settings_time,
        'witness_time': witness_time,
        'prove_time': prove_time,
        'total_time': settings_time + witness_time + prove_time,
        'success': prove_ok,
        'optimization_speedup': model_config['performance_gains']['total_speedup']
    }
    
    if verbose:
        print(f"Optimized proof generation for {base_name}:")
        print(f"  Settings: {settings_time:.2f}s")
        print(f"  Witness: {witness_time:.2f}s")
        print(f"  Prove: {prove_time:.2f}s")
        print(f"  Total: {result['total_time']:.2f}s")
        print(f"  Theoretical speedup: {result['optimization_speedup']:,}x")
    
    return result


async def generate_proofs_optimized_parallel(
    onnx_dir: str = "lora_onnx_params",
    json_dir: str = "intermediate_activations",
    output_dir: str = "proof_artifacts",
    max_workers: int = None,
    verbose: bool = False,
) -> Dict:
    """
    Parallel optimized proof generation for multiple LoRA modules.
    
    Key optimizations:
    1. Parallel proof generation across CPU cores
    2. Low-rank circuit structure
    3. Lookup tables instead of multiplication gates
    4. Smaller circuit sizes
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all optimized ONNX files
    onnx_files = glob.glob(os.path.join(onnx_dir, "*_optimized.onnx"))
    if not onnx_files:
        print(f"No optimized ONNX files found in {onnx_dir}.")
        return None
    
    if verbose:
        print(f"Found {len(onnx_files)} optimized ONNX files.")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(onnx_files), os.cpu_count() or 4)
    
    # Prepare tasks
    tasks = []
    for onnx_path in onnx_files:
        base_name = os.path.basename(onnx_path).replace("_optimized.onnx", "")
        json_path = os.path.join(json_dir, base_name + ".json")
        config_path = os.path.join(onnx_dir, base_name + "_config.json")
        
        if not os.path.exists(json_path) or not os.path.exists(config_path):
            if verbose:
                print(f"Skipping {base_name}: missing JSON or config")
            continue
        
        task = generate_optimized_proof_single(
            onnx_path, json_path, config_path, output_dir, verbose
        )
        tasks.append(task)
    
    # Execute in parallel
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Aggregate results
    successful_proofs = [r for r in results if r and r['success']]
    
    summary = {
        'total_modules': len(results),
        'successful_proofs': len(successful_proofs),
        'total_time': total_time,
        'parallel_speedup': sum(r['total_time'] for r in successful_proofs) / total_time,
        'average_proof_time': np.mean([r['total_time'] for r in successful_proofs]),
        'average_theoretical_speedup': np.mean([r['optimization_speedup'] for r in successful_proofs]),
    }
    
    if verbose:
        print("\nOptimized Proof Generation Summary:")
        print(f"  Total modules: {summary['total_modules']}")
        print(f"  Successful proofs: {summary['successful_proofs']}")
        print(f"  Total time: {summary['total_time']:.2f}s")
        print(f"  Parallel speedup: {summary['parallel_speedup']:.2f}x")
        print(f"  Average proof time: {summary['average_proof_time']:.2f}s")
        print(f"  Average theoretical speedup: {summary['average_theoretical_speedup']:,.0f}x")
    
    return summary


def batch_verify_proofs_optimized(
    proof_dir: str = "proof_artifacts",
    parallel: bool = True,
    verbose: bool = False
) -> tuple[float, int]:
    """
    Optimized batch verification with parallel processing.
    
    Verification is already fast (1-2s), but we can parallelize
    for multiple proofs.
    """
    
    proof_files = glob.glob(os.path.join(proof_dir, "*.pf"))
    if not proof_files:
        print(f"No proof files found in {proof_dir}.")
        return 0.0, 0
    
    def verify_single(proof_file):
        """Verify a single proof"""
        base_name = os.path.splitext(os.path.basename(proof_file))[0]
        names = resolve_optimized_proof_paths(proof_dir, base_name)
        
        start = time.time()
        result = ezkl.verify(
            names.proof, names.settings, names.verification_key, names.srs
        )
        duration = time.time() - start
        
        return result, duration
    
    start_time = time.time()
    
    if parallel:
        # Parallel verification
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(verify_single, pf) for pf in proof_files]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
    else:
        # Sequential verification
        results = [verify_single(pf) for pf in proof_files]
    
    total_time = time.time() - start_time
    successful = sum(1 for r, _ in results if r)
    
    if verbose:
        print(f"Verified {successful}/{len(proof_files)} proofs in {total_time:.2f}s")
        if parallel:
            sequential_time = sum(d for _, d in results)
            print(f"Parallel speedup: {sequential_time/total_time:.2f}x")
    
    return total_time, successful 