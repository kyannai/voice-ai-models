> make check-tokenizer MODEL_NAME=nvidia/parakeet-tdt-1.1b
[NeMo I 2026-01-21 22:35:01 save_restore_connector:284] Model EncDecRNNTBPEModel was successfully restored from /Users/kyan/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/6d590f77001d318fb17a0b5bf7ee329a91b52598/parakeet-tdt-0.6b-v3.nemo.
============================================================
Tokenizer type: SentencePieceTokenizer
Vocab size: 8192
============================================================

First 20 vocab entries:
      0: ' ⁇ '
      1: '<|nospeech|>'
      2: '<pad>'
      3: '<|endoftext|>'
      4: '<|startoftranscript|>'
      5: '<|pnc|>'
      6: '<|nopnc|>'
      7: '<|startofcontext|>'
      8: '<|itn|>'
      9: '<|noitn|>'
     10: '<|timestamp|>'
     11: '<|notimestamp|>'
     12: '<|diarize|>'
     13: '<|nodiarize|>'
     14: '<|spkchange|>'
     15: '<|audioseparator|>'
     16: '<|emo:undefined|>'
     17: '<|emo:neutral|>'
     18: '<|emo:happy|>'
     19: '<|emo:sad|>'

Language Support Test:
------------------------------------------------------------

Input:   'hello world'
Tokens:  [303, 3164, 2088, 2493]
Decoded: 'hello world'
Status:  ✓ Supported

Input:   '你好世界'
Tokens:  [7863, 0]
Decoded: ' ⁇ '
Status:  ✗ NOT supported (UNK tokens)

Input:   '今天天气很好'
Tokens:  [7863, 0]
Decoded: ' ⁇ '
Status:  ✗ NOT supported (UNK tokens)

Input:   'saya makan nasi'
Tokens:  [498, 5023, 2604, 284, 1415, 7866]
Decoded: 'saya makan nasi'
Status:  ✓ Supported

============================================================




> make check-tokenizer MODEL_NAME=nvidia/parakeet-tdt-1.1b
[NeMo I 2026-01-21 22:38:17 save_restore_connector:284] Model EncDecRNNTBPEModel was successfully restored from /Users/kyan/.cache/huggingface/hub/models--nvidia--parakeet-tdt-1.1b/snapshots/53276c6469d1f17a1352e30c4d11be3d0d7e9575/parakeet-tdt-1.1b.nemo.
============================================================
Tokenizer type: SentencePieceTokenizer
Vocab size: 1024
============================================================

First 20 vocab entries:
      0: ' ⁇ '
      1: 't'
      2: 'th'
      3: 'a'
      4: 'i'
      5: 'the'
      6: 're'
      7: 'w'
      8: 's'
      9: 'o'
     10: 'in'
     11: 'at'
     12: 'er'
     13: 'ou'
     14: 'nd'
     15: 'c'
     16: 'b'
     17: 'h'
     18: 'on'
     19: 'm'

Language Support Test:
------------------------------------------------------------

Input:   'hello world'
Tokens:  [67, 30, 1000, 575]
Decoded: 'hello world'
Status:  ✓ Supported

Input:   '你好世界'
Tokens:  [996, 0]
Decoded: ' ⁇ '
Status:  ✗ NOT supported (UNK tokens)

Input:   '今天天气很好'
Tokens:  [996, 0]
Decoded: ' ⁇ '
Status:  ✗ NOT supported (UNK tokens)

Input:   'saya makan nasi'
Tokens:  [279, 999, 178, 1018, 29, 42, 43, 1001]
Decoded: 'saya makan nasi'
Status:  ✓ Supported

============================================================