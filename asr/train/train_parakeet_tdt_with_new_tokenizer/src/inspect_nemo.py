#!/usr/bin/env python3
"""Inspect .nemo archive contents."""
import tarfile
import tempfile
from pathlib import Path
import torch
import yaml
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else './models/parakeet-tdt-0.6b-multilingual-init.nemo'

with tempfile.TemporaryDirectory() as tmpdir:
    with tarfile.open(model_path, 'r:*') as tar:
        tar.extractall(tmpdir)
    
    print('Archive contents:')
    for f in sorted(Path(tmpdir).iterdir()):
        print(f'  {f.name} ({f.stat().st_size} bytes)')
    
    # Check YAML config
    yaml_path = Path(tmpdir) / 'model_config.yaml'
    if yaml_path.exists():
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        print('\nmodel_config.yaml:')
        print(f"  joint.num_classes: {config.get('joint', {}).get('num_classes', 'NOT SET')}")
        print(f"  decoder.vocab_size: {config.get('decoder', {}).get('vocab_size', 'NOT SET')}")
    
    # Check checkpoint
    ckpt_path = Path(tmpdir) / 'model_weights.ckpt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print('\nCheckpoint top-level keys:')
        for k in ckpt.keys():
            print(f'  {k}')
        
        if 'hyper_parameters' in ckpt:
            hp = ckpt['hyper_parameters']
            print('\nhyper_parameters keys:')
            for k in hp.keys():
                print(f'  {k}')
            
            if 'cfg' in hp:
                cfg = hp['cfg']
                print('\nhyper_parameters.cfg.joint:')
                if 'joint' in cfg:
                    for k, v in cfg['joint'].items():
                        if not isinstance(v, (dict, list)):
                            print(f'  {k}: {v}')
                
                print('\nhyper_parameters.cfg.decoder:')
                if 'decoder' in cfg:
                    for k, v in cfg['decoder'].items():
                        if not isinstance(v, (dict, list)):
                            print(f'  {k}: {v}')
