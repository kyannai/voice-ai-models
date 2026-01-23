#!/usr/bin/env python3
"""Quick check for config mismatch."""
import warnings
warnings.filterwarnings('ignore')

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
    './models/parakeet-tdt-0.6b-multilingual-init.nemo', 
    map_location='cpu'
)

print('=== CRITICAL CONFIG CHECK ===')
print(f'Tokenizer vocab: {model.tokenizer.vocab_size}')
print(f'Decoder embed size: {model.decoder.prediction.embed.weight.shape[0]}')

# Get joint output size
for layer in model.joint.joint_net:
    if hasattr(layer, 'out_features'):
        joint_out = layer.out_features
print(f'Joint output size: {joint_out}')
print()
print('CONFIG VALUES:')
print(f'  cfg.joint.num_classes: {model.cfg.joint.num_classes}')
print(f'  cfg.decoder.vocab_size: {model.cfg.decoder.vocab_size}')
print()

# Expected values
vocab = model.tokenizer.vocab_size
expected_decoder = vocab + 1  # +1 for blank
expected_joint = vocab + 5 + 1  # +5 durations +1 blank

print('EXPECTED VALUES:')
print(f'  Decoder embed: {expected_decoder} (vocab + blank)')
print(f'  Joint output: {expected_joint} (vocab + 5 durations + blank)')
print()

# Check for mismatch
if model.cfg.joint.num_classes != expected_joint:
    print(f'!!! MISMATCH: cfg.joint.num_classes is {model.cfg.joint.num_classes} but should be {expected_joint}')
else:
    print('OK: joint config matches')
