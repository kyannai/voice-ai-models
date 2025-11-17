# Cleaning Qwen Model Predictions

## Problem

Qwen Audio models (Qwen2.5-Omni, Qwen3-Omni) sometimes output preambles before the actual transcription, such as:
- "The audio says: '...'"
- "The audio transcription is: '...'"
- "The original content of this audio is: '...'"
- etc.

These preambles need to be removed to get clean transcriptions for evaluation.

## Solution

We've implemented a two-part solution:

### 1. Prevention (for new transcriptions)

The `clean_qwen_output()` function in `transcribe/utils.py` has been updated to handle all common preamble patterns, including:
- Patterns with quotes: `The audio says: '...'`
- Patterns without quotes: `The audio says: ...`
- Incomplete patterns (missing closing quote)

This function is automatically called during transcription in:
- `transcribe_qwen25omni.py`
- `transcribe_qwen3omni.py` (if exists)
- Other Qwen-based transcription scripts

### 2. Post-processing (for existing predictions)

The `clean_predictions.py` script can clean existing prediction files:

```bash
# Clean a single file
python clean_predictions.py path/to/predictions.csv
python clean_predictions.py path/to/predictions.json

# Clean all prediction files in a directory
python clean_predictions.py path/to/outputs/

# Clean recursively
python clean_predictions.py path/to/outputs/ --recursive

# Skip backup creation
python clean_predictions.py path/to/predictions.csv --no-backup
```

## Usage Examples

### Clean a specific output directory
```bash
python clean_predictions.py outputs/Qwen2.5-Omni_Qwen2.5-Omni-7B_meso-malaya-test_auto_20251111_042149/predictions.csv
```

### Clean all predictions in outputs directory
```bash
python clean_predictions.py outputs/ --recursive
```

### Verify cleaning worked
```bash
# Compare before (backup) and after
diff predictions.csv.bak predictions.csv
```

## Features

- **Automatic backup**: Creates `.bak` files before cleaning (unless `--no-backup` is specified)
- **Safe**: Only modifies the `hypothesis` column in prediction files
- **Reports**: Shows how many hypotheses were cleaned
- **Handles both formats**: Works with both CSV and JSON prediction files

## Results

When cleaning the Qwen2.5-Omni predictions from November 11, 2025:
- **280 out of 765** (36.6%) hypotheses had preambles that were cleaned
- Both CSV and JSON files were updated
- Backup files were created for safety

## Patterns Handled

The cleaning function handles these patterns (case-insensitive):

1. `The audio says: '...'`
2. `The audio transcription is: '...'`
3. `The original content of this audio is: '...'`
4. `The transcription of audio is: '...'`
5. `The audio content is: '...'`
6. `Audio transcription: '...'`
7. Same patterns without quotes
8. Same patterns with incomplete quotes (missing closing quote)
9. Text surrounded only by quotes

## Future Improvements

If new preamble patterns are discovered, add them to the `patterns` list in `transcribe/utils.py::clean_qwen_output()`.

