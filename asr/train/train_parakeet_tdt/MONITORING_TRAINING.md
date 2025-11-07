# Monitoring Parakeet TDT Training

## Where to Find WER During Training

### 1. Console Output (During Training)

NeMo automatically prints validation metrics. Look for lines like this:

```
[NeMo I 2024-11-06 10:30:45 exp_manager:123] Validation WER: 0.1234
Epoch 0:  45%|████▌     | 4500/10000 [1:23:45<1:42:12, train_loss=0.234, val_wer=0.123]
```

**Key indicators:**
- `Validation` or `val_wer` in output
- Progress bar shows `val_wer=X.XXX`
- Logged every `eval_steps` (1000 steps for 100k config, 10000 for full dataset)

### 2. TensorBoard (Real-time Monitoring) ⭐ RECOMMENDED

**Best way to monitor training!**

#### Start TensorBoard:
```bash
# In a separate terminal (while training is running)
cd /path/to/train_parakeet_tdt
tensorboard --logdir ./outputs/parakeet-tdt-malay-asr
```

Then open: http://localhost:6006

#### What You'll See:

**Scalars Tab:**
- `val_wer` - Word Error Rate (lower is better)
- `val_loss` - Validation loss
- `train_loss_epoch` - Training loss per epoch
- `lr` - Learning rate schedule

**Example:**
```
Step 1000:  val_wer = 0.45  (45% error rate)
Step 2000:  val_wer = 0.32  (32% error rate) ✅ Improving
Step 3000:  val_wer = 0.28  (28% error rate) ✅ Still improving
Step 4000:  val_wer = 0.25  (25% error rate) ✅ Good progress
```

**Target WER for Malaysian ASR:**
- Initial (step 0): ~80-100% (model hasn't learned yet)
- After 1000 steps: ~40-60%
- After 5000 steps: ~20-30%
- Final (100k samples): ~10-15% (good)
- Final (5.2M samples): ~5-8% (excellent)

### 3. Log Files (After Training)

Check the experiment logs:
```bash
cd outputs/parakeet-tdt-malay-asr
ls -la parakeet-tdt-malay-finetuning/*/
```

**Files to check:**
- `lightning_logs.txt` - Full training log
- `nemo_log_*.txt` - NeMo-specific logs
- `events.out.tfevents.*` - TensorBoard events (binary)

**Search for WER:**
```bash
grep -i "val_wer\|validation.*wer" outputs/*/parakeet-tdt-malay-finetuning/*/lightning_logs.txt
```

### 4. Checkpoint Filenames

Checkpoints include WER in the filename:
```
outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/.../checkpoints/
  parakeet-tdt--epoch=00-step=1000-val_wer=0.4523.ckpt
  parakeet-tdt--epoch=00-step=2000-val_wer=0.3214.ckpt
  parakeet-tdt--epoch=00-step=3000-val_wer=0.2876.ckpt
```

The `val_wer=0.XXXX` shows the WER at that checkpoint!

---

## Quick Reference: What's Normal?

### Console Output During Training

```bash
# You should see these patterns:

# 1. Training progress (every 50 steps)
[NeMo I ...] Step: 50   Loss: 2.345   LR: 0.0002
[NeMo I ...] Step: 100  Loss: 1.987   LR: 0.0002
[NeMo I ...] Step: 150  Loss: 1.654   LR: 0.0002

# 2. Validation runs (every 1000 steps for 100k config)
[NeMo I ...] Validation started at step 1000
[NeMo I ...] Computing WER on validation set...
[NeMo I ...] Validation WER: 0.4523 (45.23%)
[NeMo I ...] Validation Loss: 0.8765

# 3. PyTorch Lightning progress bar
Epoch 0: 100%|██████████| 12500/12500 [2:00:00<00:00, train_loss=0.234, val_wer=0.123]
```

### TensorBoard Graphs

**WER Graph (val_wer):**
```
1.0 |*
    |  *
0.8 |    *
    |      *
0.6 |        *
    |          *
0.4 |            *___
    |                *___
0.2 |                    *___
    |                        *___
0.0 +-------------------------->
    0   2k  4k  6k  8k  10k  steps
```
↑ Should be decreasing! If it plateaus or increases, there might be an issue.

**Loss Graph (train_loss):**
```
3.0 |*
    |  *
2.5 |    *
    |      *
2.0 |        *
    |          *___
1.5 |              *___
    |                  *___
1.0 |                      *___
    |                          *___
0.5 +--------------------------->
    0   2k  4k  6k  8k  10k  steps
```
↑ Should be smoothly decreasing!

---

## Common Issues & Solutions

### Issue 1: "I don't see any WER in console output"

**Possible causes:**
1. Validation hasn't run yet (happens every `eval_steps`)
2. Output is scrolling too fast
3. NeMo logger level is too high

**Solutions:**

**A. Check when next validation will run:**
```yaml
# In config.yaml:
eval_steps: 1000  # Validation runs every 1000 steps

# If you have 100k samples, batch=8, grad_accum=16:
# Steps per epoch = 100k / (8*16) = ~781 steps
# So first validation is at step 1000 (after first epoch)
```

**B. Increase logging verbosity:**
Add to beginning of training script or set environment variable:
```bash
export NEMO_LOG_LEVEL=INFO  # or DEBUG for more detail
```

**C. Use TensorBoard instead** (more reliable):
```bash
tensorboard --logdir ./outputs/parakeet-tdt-malay-asr
```

### Issue 2: "TensorBoard shows no data"

**Check:**
```bash
# Verify TensorBoard files exist
ls outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/events.out.tfevents.*

# If files exist, restart TensorBoard with correct path:
tensorboard --logdir ./outputs --reload_interval 5
```

### Issue 3: "WER is stuck or increasing"

**This could indicate:**
- Learning rate too high → Reduce to 1e-4
- Bad data → Check data quality
- Model divergence → Restart with lower learning rate

**Check:**
```bash
# Look at full training log
tail -f outputs/*/parakeet-tdt-malay-finetuning/*/nemo_log_*.txt
```

### Issue 4: "Training seems to be running but no logs"

**Force flush logs:**
```python
# Add to train_parakeet_tdt.py at line 480, before trainer.fit():
import sys
logging.info("Starting training... (WER will be logged every 1000 steps)")
sys.stdout.flush()
```

---

## Enhanced Monitoring Setup

### Option 1: Use TensorBoard (Recommended)

**Start training:**
```bash
cd /path/to/train_parakeet_tdt
bash run_training.sh
```

**In a separate terminal:**
```bash
cd /path/to/train_parakeet_tdt
tensorboard --logdir ./outputs --reload_interval 5
```

**Open browser:** http://localhost:6006

**What to watch:**
- `val_wer` should decrease over time
- `train_loss` should decrease smoothly
- `lr` shows learning rate schedule

### Option 2: Watch Log File

```bash
cd /path/to/train_parakeet_tdt

# Start training in background
bash run_training.sh &

# Watch logs in real-time
tail -f outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/nemo_log_*.txt | grep -i "wer\|validation"
```

### Option 3: Parse Logs for WER

Create a simple script:
```bash
# save as watch_wer.sh
#!/bin/bash
while true; do
    clear
    echo "=== Latest WER Values ==="
    grep -i "val_wer" outputs/*/parakeet-tdt-malay-finetuning/*/lightning_logs.txt | tail -10
    sleep 10
done
```

Run:
```bash
chmod +x watch_wer.sh
./watch_wer.sh
```

---

## What Validation Output Looks Like

### Minimal Output (What You Might Be Seeing)
```
Epoch 0:  10%|█         | 1000/10000 [00:15:23<2:18:45, train_loss=1.234]
```

### Full Output (What NeMo Actually Logs)
```
[NeMo I 2024-11-06 10:30:45 exp_manager:567] Step 1000: Starting validation
[NeMo I 2024-11-06 10:30:45 asr_model:234] Validation DataLoader 0:   0%|          | 0/100
[NeMo I 2024-11-06 10:31:10 asr_model:234] Validation DataLoader 0: 100%|██████████| 100/100
[NeMo I 2024-11-06 10:31:10 asr_model:345] Validation epoch 0 batch 100/100
[NeMo I 2024-11-06 10:31:10 asr_model:456] val_loss: 0.8765
[NeMo I 2024-11-06 10:31:10 asr_model:457] val_wer: 0.4523
[NeMo I 2024-11-06 10:31:10 exp_manager:568] Validation complete
Epoch 0:  10%|█         | 1000/10000 [00:15:48<2:18:21, train_loss=1.234, val_wer=0.452]
```

**If you're only seeing the minimal output, the full logs are in:**
- TensorBoard (best option)
- `outputs/.../lightning_logs.txt`
- `outputs/.../nemo_log_*.txt`

---

## Expected WER Values

### For 100k Sample Training

| Step | Expected WER | Status |
|------|--------------|--------|
| 0 | 0.90-1.00 | Baseline (random) |
| 1,000 | 0.40-0.60 | Learning started |
| 2,000 | 0.30-0.40 | Good progress |
| 3,000 | 0.25-0.35 | Improving |
| 5,000 | 0.20-0.30 | Decent |
| 8,000 | 0.15-0.25 | Good |
| 10,000+ (final) | 0.10-0.20 | Very good |

### For Full Dataset (5.2M samples)

| Step | Expected WER | Status |
|------|--------------|--------|
| 0 | 0.90-1.00 | Baseline |
| 5,000 | 0.30-0.50 | Early learning |
| 10,000 | 0.20-0.35 | Progressing |
| 20,000 | 0.12-0.20 | Good |
| 50,000 | 0.08-0.15 | Very good |
| 100,000+ | 0.05-0.10 | Excellent |

**Note:** These are rough estimates. Actual values depend on:
- Data quality
- Language complexity
- Domain match
- Model configuration

---

## TL;DR: Best Way to Monitor

1. **Start training:**
   ```bash
   bash run_training.sh
   ```

2. **Open TensorBoard** (separate terminal):
   ```bash
   tensorboard --logdir ./outputs
   ```

3. **Go to:** http://localhost:6006

4. **Watch the `val_wer` graph** - Should go down! 📉

5. **Console shows minimal info** - That's normal! Use TensorBoard for details.

---

## Pro Tips

### Tip 1: SSH Port Forwarding for Remote Training

If training on remote GPU server:
```bash
# On your local machine:
ssh -L 6006:localhost:6006 user@remote-gpu-server

# On remote server:
tensorboard --logdir /path/to/outputs

# Then open on your local machine:
# http://localhost:6006
```

### Tip 2: Multiple TensorBoard Runs

Compare different experiments:
```bash
tensorboard --logdir ./outputs --reload_interval 5
```

All runs will show up in the same dashboard for easy comparison!

### Tip 3: Export WER to CSV

After training:
```bash
python -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/version_0')
ea.Reload()
wer = ea.Scalars('val_wer')
for w in wer:
    print(f'{w.step},{w.value}')
" > wer_history.csv
```

### Tip 4: Quick Status Check

```bash
# Check latest WER without TensorBoard
grep "val_wer" outputs/*/parakeet-tdt-malay-finetuning/*/lightning_logs.txt | tail -1
```

---

## Summary

✅ **WER IS being logged** - NeMo does this automatically
✅ **Best way to view:** TensorBoard (http://localhost:6006)
✅ **Also in:** Console output (look for `val_wer=X.XXX`)
✅ **Also in:** Log files (`lightning_logs.txt`)
✅ **Also in:** Checkpoint filenames

**Most users miss it because:**
- Console output scrolls fast
- WER only logs every `eval_steps` (1000 for 100k config)
- TensorBoard is the better interface but not always obvious

**Start TensorBoard first, then watch training there!** 📊

