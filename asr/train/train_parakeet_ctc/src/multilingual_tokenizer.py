#!/usr/bin/env python3
"""
Multilingual tokenizer wrapper for Parakeet CTC.

This tokenizer:
1. Uses the original SentencePiece model for Latin text (English/Malay)
2. Maps Chinese characters to their new token IDs
3. Handles hybrid text (mixed Latin + Chinese)

Usage:
    from multilingual_tokenizer import MultilingualTokenizer
    
    tokenizer = MultilingualTokenizer(
        spm_model_path="tokenizer.model",
        char_mapping_path="chinese_char_mapping.json"
    )
    
    tokens = tokenizer.encode("Hello 你好")
    text = tokenizer.decode(tokens)
"""

import json
import re
from pathlib import Path
from typing import List, Union

import sentencepiece as spm


class MultilingualTokenizer:
    """Tokenizer that combines SentencePiece with character-level Chinese."""
    
    def __init__(
        self,
        spm_model_path: str,
        char_mapping_path: str = None,
        unk_id: int = 0
    ):
        """
        Initialize the multilingual tokenizer.
        
        Args:
            spm_model_path: Path to SentencePiece .model file
            char_mapping_path: Path to JSON file with Chinese char -> ID mapping
            unk_id: ID for unknown tokens
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        
        self.spm_vocab_size = self.sp.get_piece_size()
        self.unk_id = unk_id
        
        # Load Chinese character mapping
        self.char_to_id = {}
        self.id_to_char = {}
        self.new_vocab_size = self.spm_vocab_size
        
        if char_mapping_path and Path(char_mapping_path).exists():
            with open(char_mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.char_to_id = data.get('char_to_id', {})
            self.id_to_char = {v: k for k, v in self.char_to_id.items()}
            self.new_vocab_size = data.get('new_vocab_size', self.spm_vocab_size)
            self.new_chars = data.get('new_chars', [])
        
        # Regex to identify Chinese characters
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return self.new_vocab_size
    
    def _is_chinese(self, char: str) -> bool:
        """Check if a character is Chinese."""
        return bool(self.chinese_pattern.match(char))
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        For Latin text: use SentencePiece
        For Chinese: use character-level mapping
        """
        if not text:
            return []
        
        tokens = []
        buffer = ""
        
        for char in text:
            if self._is_chinese(char):
                # Flush Latin buffer
                if buffer:
                    tokens.extend(self.sp.encode(buffer))
                    buffer = ""
                
                # Add Chinese character
                if char in self.char_to_id:
                    tokens.append(self.char_to_id[char])
                else:
                    tokens.append(self.unk_id)
            else:
                buffer += char
        
        # Flush remaining buffer
        if buffer:
            tokens.extend(self.sp.encode(buffer))
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        """
        if not ids:
            return ""
        
        result = []
        sp_buffer = []
        
        for id in ids:
            if id < self.spm_vocab_size:
                sp_buffer.append(id)
            else:
                # Flush SentencePiece buffer
                if sp_buffer:
                    result.append(self.sp.decode(sp_buffer))
                    sp_buffer = []
                
                # Decode Chinese character
                if id in self.id_to_char:
                    result.append(self.id_to_char[id])
                else:
                    result.append('�')  # Unknown
        
        # Flush remaining buffer
        if sp_buffer:
            result.append(self.sp.decode(sp_buffer))
        
        return ''.join(result)
    
    def id_to_piece(self, id: int) -> str:
        """Get the token string for an ID."""
        if id < self.spm_vocab_size:
            return self.sp.id_to_piece(id)
        elif id in self.id_to_char:
            return self.id_to_char[id]
        else:
            return '<unk>'
    
    def piece_to_id(self, piece: str) -> int:
        """Get the ID for a token string."""
        if piece in self.char_to_id:
            return self.char_to_id[piece]
        else:
            return self.sp.piece_to_id(piece)


def patch_model_tokenizer(model, char_mapping_path: str):
    """
    Patch a loaded NeMo model to use the multilingual tokenizer.
    
    Args:
        model: NeMo ASR model
        char_mapping_path: Path to Chinese character mapping JSON
    """
    # Get the original tokenizer's model path
    original_tokenizer = model.tokenizer
    
    # Create wrapper
    class TokenizerWrapper:
        def __init__(self, original, char_mapping_path):
            self._original = original
            self.tokenizer = MultilingualTokenizer(
                spm_model_path=original.tokenizer.model_path,
                char_mapping_path=char_mapping_path
            )
        
        @property
        def vocab_size(self):
            return self.tokenizer.vocab_size
        
        def text_to_ids(self, text):
            return self.tokenizer.encode(text)
        
        def ids_to_text(self, ids):
            return self.tokenizer.decode(ids)
        
        def __getattr__(self, name):
            return getattr(self._original, name)
    
    model.tokenizer = TokenizerWrapper(original_tokenizer, char_mapping_path)
    return model


if __name__ == "__main__":
    # Test the tokenizer
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-model", required=True)
    parser.add_argument("--char-mapping", required=True)
    parser.add_argument("--text", default="Hello 你好 Selamat pagi 谢谢")
    args = parser.parse_args()
    
    tokenizer = MultilingualTokenizer(args.spm_model, args.char_mapping)
    
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"SPM vocab: {tokenizer.spm_vocab_size}")
    print(f"Chinese chars: {len(tokenizer.char_to_id)}")
    print()
    
    text = args.text
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Input: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
