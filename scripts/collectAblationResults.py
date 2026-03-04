#!/usr/bin/env python3
"""Collect ablation study results and calculate mean ± std for F1 and accuracy across seeds."""

import json
import numpy as np
from pathlib import Path

base_path = Path(r'results\ablations')
datasets = ['ucihar', 'motionsense', 'wisdm', 'pamap2', 'opportunity', 'unimib', 'skoda', 'daphnet']

def get_unique_seeds(variant_folders):
    """Get unique seeds from folder names, deduplicating by seed number."""
    seed_to_folder = {}
    for folder in variant_folders:
        # Extract seed from folder name like 'A0_seed10734_20260113_133655'
        parts = folder.name.split('_')
        seed = parts[1]  # e.g., 'seed10734'
        # Keep only one folder per seed (the latest one by timestamp)
        if seed not in seed_to_folder:
            seed_to_folder[seed] = folder
        else:
            # Compare timestamps and keep the later one
            existing_ts = '_'.join(seed_to_folder[seed].name.split('_')[2:])
            new_ts = '_'.join(folder.name.split('_')[2:])
            if new_ts > existing_ts:
                seed_to_folder[seed] = folder
    return list(seed_to_folder.values())

all_results = {}

for dataset in datasets:
    arch_path = base_path / dataset / 'arch'
    if not arch_path.exists():
        print(f'Dataset {dataset} arch path not found')
        continue
    
    dataset_results = {}
    
    # Process A0-A3 variants
    for variant in ['A0', 'A1', 'A2', 'A3']:
        variant_folders = list(arch_path.glob(f'{variant}_seed*'))
        if not variant_folders:
            continue
        
        # Get unique seed folders
        unique_folders = get_unique_seeds(variant_folders)
        
        f1_scores = []
        acc_scores = []
        params = None
        macs = None
        
        for folder in unique_folders:
            result_file = folder / 'result.json'
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    final = data.get('final', {})
                    if 'f1' in final:
                        f1_scores.append(final['f1'])
                    if 'accuracy' in final:
                        acc_scores.append(final['accuracy'])
                    if params is None and 'params' in data:
                        params = data['params']
                    if macs is None and 'macs' in data:
                        macs = data['macs']
                except Exception as e:
                    print(f'Error reading {result_file}: {e}')
        
        if f1_scores:
            dataset_results[variant] = {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'acc_mean': np.mean(acc_scores) if acc_scores else 0,
                'acc_std': np.std(acc_scores) if acc_scores else 0,
                'params': params,
                'macs': macs,
                'n_seeds': len(f1_scores)
            }
    
    # Process A4_evalonly
    a4_folders = list(arch_path.glob('A4_evalonly*'))
    if a4_folders:
        # Take the latest one
        a4_folder = sorted(a4_folders)[-1]
        result_file = a4_folder / 'result.json'
        if result_file.exists():
            try:
                with open(result_file) as f:
                    data = json.load(f)
                final = data.get('final', {})
                dataset_results['A4'] = {
                    'f1_mean': final.get('f1', 0),
                    'f1_std': 0,
                    'acc_mean': final.get('accuracy', 0),
                    'acc_std': 0,
                    'params': data.get('params'),
                    'macs': data.get('macs'),
                    'n_seeds': 1
                }
            except Exception as e:
                print(f'Error reading A4 {result_file}: {e}')
    
    all_results[dataset] = dataset_results

# Print results
for dataset in datasets:
    if dataset not in all_results:
        print(f'\n### Dataset: {dataset.upper()} - NOT FOUND\n')
        continue
    
    print(f'\n### Dataset: {dataset.upper()}\n')
    print('| Variant | F1 (mean ± std) | Accuracy (mean ± std) | Params | MACs | Seeds |')
    print('|---------|-----------------|----------------------|--------|------|-------|')
    
    for variant in ['A0', 'A1', 'A2', 'A3', 'A4']:
        if variant in all_results[dataset]:
            r = all_results[dataset][variant]
            f1_str = f"{r['f1_mean']*100:.2f} ± {r['f1_std']*100:.2f}"
            acc_str = f"{r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f}"
            params_str = f"{r['params']:,}" if r['params'] else 'N/A'
            macs_str = f"{r['macs']:,}" if r['macs'] else 'N/A'
            print(f'| {variant} | {f1_str} | {acc_str} | {params_str} | {macs_str} | {r["n_seeds"]} |')
