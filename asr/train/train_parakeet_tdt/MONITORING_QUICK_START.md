# Quick Start: Monitoring WER During Training

## TL;DR

WER **is printed** during training, but **TensorBoard is the best way** to see it clearly.

## 🚀 Quick Setup (2 Steps)

### Terminal 1: Start Training
```bash
cd /path/to/train_parakeet_tdt
bash run_training.sh
```

### Terminal 2: Start TensorBoard
```bash
cd /path/to/train_parakeet_tdt
bash start_tensorboard.sh
```

**Then open:** http://localhost:6006

**Look for:** `val_wer` graph (should go DOWN 📉)

---

## Where WER Appears

### ✅ Option 1: TensorBoard (BEST)
```bash
bash start_tensorboard.sh
```
Then go to http://localhost:6006 → Scalars → `val_wer`

**Pros:**
- ✅ Real-time graphs
- ✅ Easy to see trends
- ✅ Compare multiple runs
- ✅ Professional visualization

### ✅ Option 2: Console Output
During training, look for:
```
Epoch 0:  10%|█ | 1000/10000 [00:15:48<2:18:21, train_loss=1.234, val_wer=0.452]
                                                                      ^^^^^^^^^^
                                                                      HERE!
```

**When it appears:**
- Every `eval_steps` (1000 for 100k config, 10000 for full dataset)
- For 100k training: First WER appears after ~15 minutes

**Pros:**
- ✅ No extra tools needed
- ✅ Shows up automatically

**Cons:**
- ⚠️ Easy to miss in scrolling output
- ⚠️ Only shows current value, not trends

### ✅ Option 3: Checkpoint Filenames
```bash
ls outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/checkpoints/
```

Files are named: `parakeet-tdt--epoch=00-step=1000-val_wer=0.4523.ckpt`
                                                      ^^^^^^^^^^^^^^
                                                      WER in filename!

**Pros:**
- ✅ Easy to see best checkpoint
- ✅ Persists after training

**Cons:**
- ⚠️ Only for full dataset training (100k config has checkpoints disabled)

### ✅ Option 4: Log Files
```bash
grep "val_wer" outputs/*/parakeet-tdt-malay-finetuning/*/lightning_logs.txt
```

**Pros:**
- ✅ Complete history
- ✅ Can grep/parse

**Cons:**
- ⚠️ Not as visual as TensorBoard

---

## What's Normal?

### 100k Sample Training

| Time | Step | Expected WER | Status |
|------|------|--------------|--------|
| Start | 0 | 0.90-1.00 | 🔴 Baseline (random guesses) |
| ~15 min | 1,000 | 0.40-0.60 | 🟡 Learning started |
| ~30 min | 2,000 | 0.30-0.40 | 🟢 Good progress |
| ~1 hour | 4,000 | 0.20-0.30 | 🟢 Improving nicely |
| ~2 hours | 8,000+ | 0.10-0.20 | 🟢 Very good! |

### Full Dataset Training (5.2M samples)

| Time | Step | Expected WER | Status |
|------|------|--------------|--------|
| Start | 0 | 0.90-1.00 | 🔴 Baseline |
| ~2 hours | 10,000 | 0.20-0.35 | 🟡 Early learning |
| ~6 hours | 30,000 | 0.12-0.20 | 🟢 Good progress |
| ~12 hours | 60,000 | 0.08-0.15 | 🟢 Very good |
| ~24 hours | 120,000+ | 0.05-0.10 | 🟢 Excellent! |

**Note:** WER should generally **decrease over time**. If it increases or plateaus, something might be wrong.

---

## Example Console Output

### What You'll See Every ~50 Steps
```bash
[NeMo I 2024-11-06 10:15:23] Step: 50   Loss: 2.345
[NeMo I 2024-11-06 10:15:45] Step: 100  Loss: 1.987
[NeMo I 2024-11-06 10:16:08] Step: 150  Loss: 1.654
```

### What You'll See Every ~1000 Steps (Validation)
```bash
[NeMo I 2024-11-06 10:30:45] Validation started at step 1000
[NeMo I 2024-11-06 10:31:10] val_wer: 0.4523 (45.23%)  ← HERE!
[NeMo I 2024-11-06 10:31:10] val_loss: 0.8765
```

### PyTorch Lightning Progress Bar
```
Epoch 0:  10%|█         | 1000/10000 [15:48<2:18:21, train_loss=1.234, val_wer=0.452]
                                                                        ^^^^^^^^^^^^^
                                                                        WER shown here too!
```

---

## Troubleshooting

### "I don't see any WER"

**Check:**
1. Has validation run yet? (First validation at step 1000 for 100k config)
2. Is console output scrolling too fast?
3. Try TensorBoard instead: `bash start_tensorboard.sh`

**For 100k training:**
- First WER appears after ~15 minutes (at step 1000)
- If you started recently, just wait!

### "TensorBoard shows no data"

**Try:**
```bash
# Make sure training has started and created outputs
ls outputs/

# Restart TensorBoard
bash start_tensorboard.sh
```

### "WER is increasing or stuck"

**This could mean:**
- 🔴 Learning rate too high → Reduce to 1e-4
- 🔴 Data quality issues → Check manifests
- 🔴 Model divergence → Restart training

**Check training loss:**
- Should also be decreasing
- If loss is decreasing but WER isn't, wait longer

---

## Commands Cheat Sheet

### Start Everything
```bash
# Terminal 1: Training
bash run_training.sh

# Terminal 2: Monitoring
bash start_tensorboard.sh
```

### Check Latest WER
```bash
# Quick check without TensorBoard
grep "val_wer" outputs/*/parakeet-tdt-malay-finetuning/*/lightning_logs.txt | tail -5
```

### Watch Training Log Live
```bash
tail -f outputs/parakeet-tdt-malay-asr/parakeet-tdt-malay-finetuning/*/nemo_log_*.txt
```

### List All Checkpoints with WER
```bash
ls -lh outputs/*/parakeet-tdt-malay-finetuning/*/checkpoints/ | grep "val_wer"
```

---

## Summary

✅ **WER IS logged** - Every 1000 steps (100k config) or 10000 steps (full dataset)
✅ **Best way to see it:** TensorBoard (`bash start_tensorboard.sh`)
✅ **Also appears in:** Console output, log files, checkpoint names
✅ **Normal values:** Start at ~0.9, end at 0.1-0.2 (100k) or 0.05-0.1 (full dataset)
✅ **First WER appears:** After ~15 minutes (100k config)

**Don't stress if you don't see it immediately in console - use TensorBoard!** 📊

---

## Visual Guide

### TensorBoard Interface
```
┌─────────────────────────────────────────┐
│ TensorBoard - localhost:6006            │
├─────────────────────────────────────────┤
│  [Scalars] [Images] [Graphs] [Dist]    │
├─────────────────────────────────────────┤
│                                         │
│  val_wer ──────────────────────────────│
│  1.0 │*                                 │
│      │  *                               │
│  0.8 │    *                             │
│      │      *                           │
│  0.6 │        *                         │
│      │          *___                    │
│  0.4 │              *___                │
│      │                  *___            │
│  0.2 │                      *___        │
│  0.0 └──────────────────────────────>  │
│       0    2k   4k   6k   8k   10k      │
│                                         │
│  ← This graph should go DOWN!           │
└─────────────────────────────────────────┘
```

**If the graph goes down → Training is working! ✅**
**If the graph is flat or up → Something's wrong! ⚠️**

---

For detailed information, see: **MONITORING_TRAINING.md**

