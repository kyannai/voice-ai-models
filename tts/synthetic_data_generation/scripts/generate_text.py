#!/usr/bin/env python3
"""
Generate conversational text for TTS training using Gemini 2.5 Flash

Generates diverse sentences for banking customer service in:
- Malay (60%)
- English (20%)
- Code-switching (20%)

Usage:
    python generate_text.py --config ../config.yaml --output ../outputs/sentences.json
"""

import argparse
import json
import os
import random
import time
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Environment variables
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextSample:
    """Represents a generated text sample"""
    text: str
    language: str  # 'malay', 'english', 'code_switching'
    topic: str
    sentence_type: str
    speaker_id: str
    word_count: int


# Banking domain vocabulary for prompts
BANKING_CONTEXT = {
    "malay": {
        "greetings": [
            "Selamat pagi", "Selamat petang", "Selamat sejahtera",
            "Terima kasih kerana menghubungi kami"
        ],
        "account_terms": [
            "akaun simpanan", "akaun semasa", "baki akaun", "penyata akaun",
            "nombor akaun", "transaksi", "pemindahan wang", "deposit"
        ],
        "card_terms": [
            "kad kredit", "kad debit", "had kredit", "bayaran minimum",
            "aktivasi kad", "PIN", "CVV"
        ],
        "loan_terms": [
            "pinjaman peribadi", "pinjaman perumahan", "kadar faedah",
            "tempoh bayaran", "ansuran bulanan"
        ],
        "common_phrases": [
            "Boleh saya bantu", "Sila tunggu sebentar", "Maaf atas kesulitan",
            "Terima kasih atas kesabaran anda", "Ada apa-apa lagi"
        ]
    },
    "english": {
        "greetings": [
            "Good morning", "Good afternoon", "Hello",
            "Thank you for calling"
        ],
        "account_terms": [
            "savings account", "current account", "account balance",
            "bank statement", "account number", "transaction", "transfer"
        ],
        "card_terms": [
            "credit card", "debit card", "credit limit", "minimum payment",
            "card activation", "PIN number"
        ],
        "loan_terms": [
            "personal loan", "home loan", "interest rate",
            "repayment period", "monthly installment"
        ],
        "common_phrases": [
            "How may I help you", "Please hold on", "I apologize for the inconvenience",
            "Thank you for your patience", "Is there anything else"
        ]
    }
}


class GeminiTextGenerator:
    """Generate text using Gemini 2.5 Flash API"""
    
    def __init__(self, config: Dict):
        """Initialize Gemini text generator"""
        self.config = config
        self.text_config = config['text_generation']
        
        # Get API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            self.model = genai.GenerativeModel(self.text_config['model'])
            logger.info(f"Initialized Gemini model: {self.text_config['model']}")
            
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )
        
        # Load configuration
        self.samples_per_speaker = self.text_config['samples_per_speaker']
        self.language_dist = self.text_config['language_distribution']
        self.topics = self.text_config['topics']
        self.sentence_types = self.text_config['sentence_types']
        self.speakers = list(config['speakers'].keys())
        
        self.min_words = self.text_config.get('min_words_per_sample', 8)
        self.max_words = self.text_config.get('max_words_per_sample', 25)
    
    def _build_prompt(
        self,
        language: str,
        topic: str,
        sentence_type: str,
        count: int = 10
    ) -> str:
        """Build prompt for Gemini to generate sentences"""
        
        lang_instruction = {
            "malay": "Generate sentences in Bahasa Melayu (Malay language).",
            "english": "Generate sentences in English.",
            "code_switching": "Generate sentences mixing Malay and English naturally (code-switching), like Malaysian speakers do. Example: 'Ok, saya akan check your account balance sekarang ya.'"
        }
        
        topic_descriptions = {
            "greetings_closings": "greeting customers, welcoming them, or closing conversations politely",
            "account_inquiry": "checking account balance, asking about account details, explaining account information",
            "transaction_status": "confirming transfers, explaining transaction status, payment confirmations",
            "card_services": "credit/debit card activation, card limits, card issues, PIN reset",
            "loan_applications": "loan inquiries, loan application status, repayment schedules",
            "online_banking": "mobile app issues, e-banking login, TAC codes, password reset",
            "complaints_feedback": "handling complaints, apologizing, escalating issues, taking feedback"
        }
        
        type_instructions = {
            "greeting": "Start with a greeting or welcome message.",
            "confirmation": "Confirm understanding or acknowledge the customer's request.",
            "information": "Provide information about account, balance, or status.",
            "question": "Ask the customer a question to gather information.",
            "instruction": "Give the customer clear instructions on what to do.",
            "apology": "Apologize for an issue or inconvenience.",
            "closing": "End the conversation politely with thanks or well wishes."
        }
        
        prompt = f"""You are generating training data for a Malaysian banking customer service voice AI agent.

{lang_instruction.get(language, lang_instruction['malay'])}

Topic: {topic_descriptions.get(topic, topic)}
Sentence type: {type_instructions.get(sentence_type, '')}

Requirements:
- Generate exactly {count} unique sentences
- Each sentence should be {self.min_words}-{self.max_words} words
- Use natural, conversational tone suitable for phone customer service
- For banking context, use appropriate Malaysian banking terminology
- Make sentences sound natural when spoken aloud
- Vary the sentences - don't repeat the same structure

{"For code-switching, mix Malay and English naturally like Malaysians speak. Common patterns: starting in one language and switching mid-sentence, using English technical terms in Malay sentences, or mixing particles like 'lah', 'kan', 'ya'." if language == "code_switching" else ""}

Output format: Return ONLY a JSON array of strings, no other text.
Example: ["Sentence one here.", "Sentence two here."]

Generate {count} sentences:"""
        
        return prompt
    
    def generate_batch(
        self,
        language: str,
        topic: str,
        sentence_type: str,
        count: int = 10,
        retries: int = 3
    ) -> List[str]:
        """Generate a batch of sentences using Gemini"""
        
        prompt = self._build_prompt(language, topic, sentence_type, count)
        
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                
                # Parse JSON response
                # Handle potential markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                
                sentences = json.loads(text)
                
                # Validate sentences
                valid_sentences = []
                for s in sentences:
                    if isinstance(s, str) and len(s.split()) >= self.min_words:
                        valid_sentences.append(s.strip())
                
                if valid_sentences:
                    return valid_sentences
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.warning(f"Generation error (attempt {attempt + 1}): {e}")
            
            time.sleep(1)  # Rate limiting
        
        logger.error(f"Failed to generate sentences after {retries} retries")
        return []
    
    def generate_for_speaker(
        self,
        speaker_id: str,
        num_samples: int
    ) -> List[TextSample]:
        """Generate all samples for a single speaker"""
        
        samples = []
        
        # Calculate samples per language
        samples_malay = int(num_samples * self.language_dist['malay'])
        samples_english = int(num_samples * self.language_dist['english'])
        samples_codesw = num_samples - samples_malay - samples_english
        
        language_counts = {
            'malay': samples_malay,
            'english': samples_english,
            'code_switching': samples_codesw
        }
        
        logger.info(f"Generating for {speaker_id}:")
        logger.info(f"  Malay: {samples_malay}, English: {samples_english}, Code-switching: {samples_codesw}")
        
        for language, count in language_counts.items():
            if count == 0:
                continue
            
            # Distribute across topics based on weights
            topic_samples = {}
            remaining = count
            
            for topic in self.topics:
                topic_count = int(count * topic['weight'])
                topic_samples[topic['name']] = topic_count
                remaining -= topic_count
            
            # Add remaining to first topic
            if remaining > 0 and self.topics:
                topic_samples[self.topics[0]['name']] += remaining
            
            # Generate for each topic
            for topic_name, topic_count in topic_samples.items():
                if topic_count == 0:
                    continue
                
                # Distribute across sentence types
                sentences_per_type = max(1, topic_count // len(self.sentence_types))
                
                for sent_type in self.sentence_types:
                    batch_size = min(10, sentences_per_type)
                    
                    generated = self.generate_batch(
                        language=language,
                        topic=topic_name,
                        sentence_type=sent_type,
                        count=batch_size
                    )
                    
                    for text in generated:
                        samples.append(TextSample(
                            text=text,
                            language=language,
                            topic=topic_name,
                            sentence_type=sent_type,
                            speaker_id=speaker_id,
                            word_count=len(text.split())
                        ))
                    
                    # Small delay for rate limiting
                    time.sleep(0.5)
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Trim to exact count if we generated too many
        if len(samples) > num_samples:
            samples = samples[:num_samples]
        
        logger.info(f"Generated {len(samples)} samples for {speaker_id}")
        
        return samples
    
    def generate_all(self) -> Dict[str, List[TextSample]]:
        """Generate samples for all speakers"""
        
        all_samples = {}
        
        for speaker_id in tqdm(self.speakers, desc="Generating for speakers"):
            samples = self.generate_for_speaker(speaker_id, self.samples_per_speaker)
            all_samples[speaker_id] = samples
        
        return all_samples


class FallbackTextGenerator:
    """Fallback template-based generator when Gemini is not available"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.text_config = config['text_generation']
        self.speakers = list(config['speakers'].keys())
        self.samples_per_speaker = self.text_config['samples_per_speaker']
        self.language_dist = self.text_config['language_distribution']
    
    def _generate_malay_sentences(self, count: int) -> List[str]:
        """Generate Malay banking sentences from templates"""
        templates = [
            "Selamat {greeting}, terima kasih kerana menghubungi perkhidmatan pelanggan kami.",
            "Baik, saya faham. Baki akaun semasa anda ialah {amount} ringgit.",
            "Untuk pengaktifan kad kredit, sila berikan nombor kad anda.",
            "Maaf atas kesulitan yang berlaku. Kami akan selesaikan masalah ini segera.",
            "Transaksi pemindahan wang anda telah berjaya diproses.",
            "Boleh saya tahu nombor akaun anda untuk pengesahan?",
            "Sila tunggu sebentar, saya akan semak status permohonan pinjaman anda.",
            "Terima kasih atas kesabaran anda. Ada apa-apa lagi yang boleh saya bantu?",
            "Untuk reset kata laluan e-banking, sila ikut arahan yang dihantar ke telefon anda.",
            "Penyata akaun bulanan anda akan dihantar melalui emel dalam masa tiga hari bekerja.",
        ]
        
        greetings = ["pagi", "petang", "sejahtera"]
        amounts = ["lima ribu tiga ratus", "dua belas ribu", "tujuh ratus lima puluh", "tiga puluh empat ribu"]
        
        sentences = []
        for _ in range(count):
            template = random.choice(templates)
            sentence = template.format(
                greeting=random.choice(greetings),
                amount=random.choice(amounts)
            )
            sentences.append(sentence)
        
        return sentences
    
    def _generate_english_sentences(self, count: int) -> List[str]:
        """Generate English banking sentences from templates"""
        templates = [
            "Good {greeting}, thank you for calling our customer service.",
            "I understand. Your current account balance is {amount} ringgit.",
            "For credit card activation, please provide your card number.",
            "I apologize for the inconvenience. We will resolve this issue promptly.",
            "Your fund transfer has been successfully processed.",
            "May I have your account number for verification?",
            "Please hold on, I will check the status of your loan application.",
            "Thank you for your patience. Is there anything else I can help you with?",
            "To reset your e-banking password, please follow the instructions sent to your phone.",
            "Your monthly account statement will be sent via email within three working days.",
        ]
        
        greetings = ["morning", "afternoon", "evening"]
        amounts = ["five thousand three hundred", "twelve thousand", "seven hundred fifty"]
        
        sentences = []
        for _ in range(count):
            template = random.choice(templates)
            sentence = template.format(
                greeting=random.choice(greetings),
                amount=random.choice(amounts)
            )
            sentences.append(sentence)
        
        return sentences
    
    def _generate_codeswitching_sentences(self, count: int) -> List[str]:
        """Generate code-switching sentences from templates"""
        templates = [
            "Ok, saya akan check your account balance sekarang ya.",
            "Untuk activate your card, you need to call our hotline dulu.",
            "Baik, I understand your concern. Let me help you dengan masalah ini.",
            "Your transaction has been successful. Terima kasih kerana menggunakan perkhidmatan kami.",
            "Sila hold on sebentar, saya nak verify your details dulu.",
            "Sorry for the inconvenience ya. Kami akan cuba selesaikan as soon as possible.",
            "Boleh you provide your account number untuk verification purposes?",
            "Your loan application is still under review. Kami akan contact you dalam masa tiga hari.",
            "Untuk e-banking issues, you boleh juga pergi ke branch terdekat.",
            "Thank you for your patience. Ada anything else yang I boleh bantu?",
        ]
        
        sentences = []
        for _ in range(count):
            sentences.append(random.choice(templates))
        
        return sentences
    
    def generate_for_speaker(self, speaker_id: str, num_samples: int) -> List[TextSample]:
        """Generate samples for a speaker using templates"""
        
        samples = []
        
        # Calculate samples per language
        n_malay = int(num_samples * self.language_dist['malay'])
        n_english = int(num_samples * self.language_dist['english'])
        n_codesw = num_samples - n_malay - n_english
        
        # Generate sentences
        malay_sentences = self._generate_malay_sentences(n_malay)
        english_sentences = self._generate_english_sentences(n_english)
        codesw_sentences = self._generate_codeswitching_sentences(n_codesw)
        
        for text in malay_sentences:
            samples.append(TextSample(
                text=text,
                language='malay',
                topic='general',
                sentence_type='statement',
                speaker_id=speaker_id,
                word_count=len(text.split())
            ))
        
        for text in english_sentences:
            samples.append(TextSample(
                text=text,
                language='english',
                topic='general',
                sentence_type='statement',
                speaker_id=speaker_id,
                word_count=len(text.split())
            ))
        
        for text in codesw_sentences:
            samples.append(TextSample(
                text=text,
                language='code_switching',
                topic='general',
                sentence_type='statement',
                speaker_id=speaker_id,
                word_count=len(text.split())
            ))
        
        random.shuffle(samples)
        return samples[:num_samples]
    
    def generate_all(self) -> Dict[str, List[TextSample]]:
        """Generate samples for all speakers"""
        all_samples = {}
        
        for speaker_id in tqdm(self.speakers, desc="Generating for speakers"):
            samples = self.generate_for_speaker(speaker_id, self.samples_per_speaker)
            all_samples[speaker_id] = samples
        
        return all_samples


def save_samples(
    samples: Dict[str, List[TextSample]],
    output_dir: Path
):
    """Save generated samples to JSON files"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-speaker files
    for speaker_id, speaker_samples in samples.items():
        speaker_dir = output_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        samples_data = [
            {
                'text': s.text,
                'language': s.language,
                'topic': s.topic,
                'sentence_type': s.sentence_type,
                'speaker_id': s.speaker_id,
                'word_count': s.word_count
            }
            for s in speaker_samples
        ]
        
        output_file = speaker_dir / 'sentences.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(samples_data)} sentences to {output_file}")
    
    # Save combined file
    all_samples = []
    for speaker_id, speaker_samples in samples.items():
        for s in speaker_samples:
            all_samples.append({
                'text': s.text,
                'language': s.language,
                'topic': s.topic,
                'sentence_type': s.sentence_type,
                'speaker_id': s.speaker_id,
                'word_count': s.word_count
            })
    
    combined_file = output_dir / 'all_sentences.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(all_samples)} total sentences to {combined_file}")
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION STATISTICS")
    logger.info("=" * 60)
    
    total = len(all_samples)
    by_language = {}
    by_speaker = {}
    
    for s in all_samples:
        by_language[s['language']] = by_language.get(s['language'], 0) + 1
        by_speaker[s['speaker_id']] = by_speaker.get(s['speaker_id'], 0) + 1
    
    logger.info(f"\nTotal samples: {total}")
    logger.info("\nBy language:")
    for lang, count in by_language.items():
        logger.info(f"  {lang}: {count} ({count/total*100:.1f}%)")
    
    logger.info("\nBy speaker:")
    for speaker, count in by_speaker.items():
        logger.info(f"  {speaker}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text for TTS training using Gemini 2.5 Flash"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for generated sentences'
    )
    parser.add_argument(
        '--use-fallback',
        action='store_true',
        help='Use template-based fallback instead of Gemini'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Override max samples per speaker'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try parent directory
        config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override samples if specified
    if args.max_samples:
        config['text_generation']['samples_per_speaker'] = args.max_samples
    
    # Initialize generator
    if args.use_fallback:
        logger.info("Using fallback template-based generator")
        generator = FallbackTextGenerator(config)
    else:
        try:
            generator = GeminiTextGenerator(config)
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not initialize Gemini: {e}")
            logger.info("Falling back to template-based generator")
            generator = FallbackTextGenerator(config)
    
    # Generate samples
    logger.info("Starting text generation...")
    samples = generator.generate_all()
    
    # Save samples
    output_dir = Path(args.output)
    save_samples(samples, output_dir)
    
    logger.info("\n✅ Text generation completed!")


if __name__ == "__main__":
    main()

