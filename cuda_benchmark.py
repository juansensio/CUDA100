import torch
import time
import pandas as pd
from typing import Callable, Dict, List
import numpy as np

class CUDABenchmark:
    """Systematic benchmarking framework for CUDA implementations"""
    
    def __init__(self, warmup_runs=3, benchmark_runs=10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    def benchmark_function(self, 
                          func: Callable,
                          name: str,
                          *args,
                          **kwargs) -> Dict:
        """Benchmark a function with proper warmup and multiple runs"""
        # Warmup
        for _ in range(self.warmup_runs):
            _ = func(*args, **kwargs)
        
        torch.cuda.synchronize()  # Ensure warmup is complete
        
        # Actual benchmark
        times = []
        for _ in range(self.benchmark_runs):
            torch.cuda.synchronize()  # Start timing from clean state
            start = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()  # Wait for GPU to finish
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'name': name,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'result': result
        }
    
    def compare_implementations(self,
                               implementations: Dict[str, Callable],
                               test_data: Dict,
                               reference_func: Callable = None,
                               reference_name: str = "PyTorch") -> pd.DataFrame:
        """Compare multiple implementations against a reference"""
        results = []
        
        # Benchmark reference first
        if reference_func:
            ref_result = self.benchmark_function(
                reference_func, reference_name, **test_data
            )
            results.append(ref_result)
            ref_output = ref_result['result']
        
        # Benchmark all implementations
        for name, func in implementations.items():
            result = self.benchmark_function(func, name, **test_data)
            
            # Verify correctness if reference provided
            if reference_func:
                try:
                    if isinstance(ref_output, torch.Tensor) and isinstance(result['result'], torch.Tensor):
                        is_correct = torch.allclose(ref_output, result['result'], rtol=1e-5, atol=1e-6)
                        result['correct'] = is_correct
                        if not is_correct:
                            max_diff = (ref_output - result['result']).abs().max().item()
                            result['max_diff'] = max_diff
                    else:
                        result['correct'] = None
                except Exception as e:
                    result['correct'] = f"Error: {e}"
            
            results.append(result)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Add speedup column if reference exists
        if reference_func and 'mean_ms' in df.columns:
            ref_time = df[df['name'] == reference_name]['mean_ms'].values[0]
            df['speedup'] = ref_time / df['mean_ms']
            df['speedup'] = df['speedup'].fillna(1.0)
        
        return df
    
    def benchmark_parameter_sweep(self,
                                  func: Callable,
                                  name: str,
                                  param_name: str,
                                  param_values: List,
                                  **fixed_kwargs) -> pd.DataFrame:
        """Benchmark function across different parameter values"""
        results = []
        for param_val in param_values:
            kwargs = {**fixed_kwargs, param_name: param_val}
            result = self.benchmark_function(func, name, **kwargs)
            result[param_name] = param_val
            results.append(result)
        return pd.DataFrame(results)