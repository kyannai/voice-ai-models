#!/usr/bin/env python3
"""
Unified sentence generation script for synthetic training data
Generates sentences mixing Malaysian names and numbers for efficient training
All numbers are converted to Malay words to match spoken audio
"""

import json
import random
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

# Import number-to-Malay conversion functions
from number_to_malay import (
    number_to_malay_words,
    currency_to_malay,
    date_to_malay,
    phone_to_malay,
    percentage_to_malay,
    time_to_malay,
    ic_number_to_malay,
    format_amount_for_tts
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentenceGenerator:
    """Generate training sentences with Malaysian names and numbers"""
    
    def __init__(self, config_path: str):
        """Load configuration and data sources"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.load_data_sources()
        self.load_templates()
        
        logger.info("Sentence generator initialized")
    
    def load_data_sources(self):
        """Load politicians, historical figures, and number patterns"""
        # Load politicians
        with open(self.config['data_sources']['politicians_file'], 'r') as f:
            politicians_data = json.load(f)
            self.politicians = politicians_data['politicians']
        
        # Load historical figures
        with open(self.config['data_sources']['historical_file'], 'r') as f:
            historical_data = json.load(f)
            self.historical = historical_data['historical_figures']
        
        # Combine all names
        self.all_names = self.politicians + self.historical
        
        # Load number patterns
        with open(self.config['data_sources']['number_patterns_file'], 'r') as f:
            self.number_patterns = json.load(f)
        
        logger.info(f"Loaded {len(self.politicians)} politicians")
        logger.info(f"Loaded {len(self.historical)} historical figures")
        logger.info(f"Total: {len(self.all_names)} names")
    
    def load_templates(self):
        """Load sentence templates"""
        # Load mixed templates (priority)
        with open(self.config['data_sources']['mixed_templates_file'], 'r') as f:
            mixed_data = json.load(f)
            self.mixed_templates = []
            for category, templates in mixed_data['categories'].items():
                self.mixed_templates.extend(templates)
        
        # Load pure name templates (optional)
        with open(self.config['data_sources']['name_templates_file'], 'r') as f:
            name_data = json.load(f)
            self.name_templates = []
            for category, templates in name_data['categories'].items():
                self.name_templates.extend(templates)
        
        # Load pure number templates (optional)
        with open(self.config['data_sources']['number_templates_file'], 'r') as f:
            number_data = json.load(f)
            self.number_templates = []
            for category, templates in number_data['categories'].items():
                self.number_templates.extend(templates)
        
        logger.info(f"Loaded {len(self.mixed_templates)} mixed templates")
        logger.info(f"Loaded {len(self.name_templates)} name templates")
        logger.info(f"Loaded {len(self.number_templates)} number templates")
    
    def generate_currency_amount(self) -> Dict[str, str]:
        """Generate currency amount in Malay words"""
        amount_type = random.choice(['small', 'medium', 'large', 'very_large'])
        ranges = self.number_patterns['number_types']['currency']['amount_ranges']
        amount = random.randint(ranges[amount_type][0], ranges[amount_type][1])
        
        # Always use ringgit format (not RM prefix)
        use_rm = False
        
        # Convert to Malay words for transcript
        amount_words = currency_to_malay(amount, use_rm)
        
        return {
            'amount': amount_words,
            'whole_amount': amount_words
        }
    
    def generate_date(self) -> Dict[str, str]:
        """Generate Malaysian date in Malay words"""
        months = self.number_patterns['number_types']['dates']['months_malay']
        days = self.number_patterns['number_types']['dates']['days_malay']
        
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(2020, 2025)
        
        # Convert to Malay words for transcript
        date_words = date_to_malay(day, months[month-1], year)
        day_words = number_to_malay_words(day)
        
        return {
            'date': date_words,
            'day': day_words,
            'month': str(month).zfill(2),
            'month_malay': months[month-1],
            'year': number_to_malay_words(year),
            'day_name': random.choice(days)
        }
    
    def generate_phone_number(self) -> Dict[str, str]:
        """Generate Malaysian phone number in Malay words"""
        prefixes = self.number_patterns['number_types']['phone_numbers']['prefixes']
        all_prefixes = []
        for prefix_list in prefixes.values():
            all_prefixes.extend(prefix_list)
        
        prefix = random.choice(all_prefixes)
        part1 = str(random.randint(100, 999))
        part2 = str(random.randint(1000, 9999))
        
        # Convert phone number to Malay words (digit by digit)
        phone_number = f"{prefix}-{part1}-{part2}"
        phone_words = phone_to_malay(phone_number)
        
        return {
            'phone': phone_words,
            'prefix': phone_to_malay(prefix),
            'number1': phone_to_malay(part1),
            'number2': phone_to_malay(part2)
        }
    
    def generate_percentage(self) -> Dict[str, str]:
        """Generate percentage in Malay words"""
        # Generate realistic percentages
        percent = round(random.uniform(0.5, 99.5), 1)
        
        # Convert to Malay words
        percent_words = percentage_to_malay(percent)
        
        return {
            'percent': percent_words,
            'percentage': percent_words
        }
    
    def generate_number(self) -> Dict[str, str]:
        """Generate general number in Malay words"""
        number_type = random.choice(['small', 'medium', 'large', 'very_large'])
        ranges = self.number_patterns['number_types']['general_numbers']['number_ranges']
        number = random.randint(ranges[number_type][0], ranges[number_type][1])
        
        # Convert to Malay words
        number_words = number_to_malay_words(number)
        number1 = random.randint(1, 1000)
        number2 = random.randint(1, 1000)
        
        return {
            'number': number_words,
            'number1': number_to_malay_words(number1),
            'number2': number_to_malay_words(number2)
        }
    
    def generate_time(self) -> Dict[str, str]:
        """Generate time in Malay words"""
        hour = random.randint(1, 12)
        minute = random.randint(0, 59)
        period = random.choice(['pagi', 'petang', 'malam'])
        
        # Convert to Malay words
        time_words = time_to_malay(hour, minute)
        hour2 = random.randint(1, 12)
        minute2 = random.randint(0, 59)
        time2_words = time_to_malay(hour2, minute2)
        
        return {
            'time': time_words,
            'hour': number_to_malay_words(hour),
            'minute': number_to_malay_words(minute) if minute > 0 else "",
            'period': period,
            'time2': time2_words
        }
    
    def generate_ic_number(self) -> Dict[str, str]:
        """Generate IC number in Malay words (digit by digit)"""
        year = random.randint(60, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        pb = random.randint(1, 99)
        gggg = random.randint(1000, 9999)
        
        ic_number = f"{year:02d}{month:02d}{day:02d}-{pb:02d}-{gggg}"
        
        # Convert to Malay words (digit by digit)
        ic_words = ic_number_to_malay(ic_number)
        
        return {
            'ic_number': ic_words
        }
    
    def generate_all_placeholders(self) -> Dict[str, str]:
        """Generate all possible placeholders"""
        placeholders = {}
        
        # Currency
        placeholders.update(self.generate_currency_amount())
        
        # Date
        placeholders.update(self.generate_date())
        
        # Phone
        placeholders.update(self.generate_phone_number())
        
        # Percentage
        placeholders.update(self.generate_percentage())
        
        # Numbers
        placeholders.update(self.generate_number())
        
        # Time
        placeholders.update(self.generate_time())
        
        # IC
        placeholders.update(self.generate_ic_number())
        
        # Additional placeholders (convert all to Malay words)
        passport_num = random.randint(10000000, 99999999)
        ref_num = random.randint(100000, 999999)
        reg_num = random.randint(100000, 999999)
        policy_num = random.randint(1000000, 9999999)
        license_num = random.randint(100000, 999999)
        id_num = random.randint(100000, 999999)
        voucher_num = random.randint(10000, 99999)
        serial_num = random.randint(100000, 999999)
        invoice_num = random.randint(10000, 99999)
        postcode_num = random.randint(10000, 99999)
        plate_num = random.randint(1000, 9999)
        receipt_num = random.randint(10000, 99999)
        order_num = random.randint(10000, 99999)
        account_num = random.randint(1000000000, 9999999999)
        
        plate_letter = random.choice(['A', 'B', 'C', 'D'])
        
        placeholders.update({
            'passport': f"A {phone_to_malay(str(passport_num))}",
            'ref_number': f"REF {phone_to_malay(str(ref_num))}",
            'reg_number': f"SSM {phone_to_malay(str(reg_num))}",
            'policy_number': f"POL {phone_to_malay(str(policy_num))}",
            'license_number': f"LIC {phone_to_malay(str(license_num))}",
            'id_number': f"ID {phone_to_malay(str(id_num))}",
            'voucher_number': f"VOU {phone_to_malay(str(voucher_num))}",
            'serial_number': f"SN {phone_to_malay(str(serial_num))}",
            'invoice_number': f"INV {phone_to_malay(str(invoice_num))}",
            'postcode': phone_to_malay(str(postcode_num)),
            'plate_number': f"W {plate_letter} {phone_to_malay(str(plate_num))}",
            'receipt_number': f"REC {phone_to_malay(str(receipt_num))}",
            'order_number': f"ORD {phone_to_malay(str(order_num))}",
            'account_number': phone_to_malay(str(account_num))
        })
        
        return placeholders
    
    def fill_template(self, template: str, person: Dict, placeholders: Dict) -> str:
        """Fill a template with person and number data"""
        # Start with person data
        values = {
            'name': person.get('name', ''),
            'honorific': person.get('honorific', ''),
            'position': person.get('position', person.get('title', '')),
            'party': person.get('party', ''),
        }
        
        # Add all number placeholders
        values.update(placeholders)
        
        # Fill template
        try:
            sentence = template.format(**values)
            return sentence
        except KeyError as e:
            logger.warning(f"Missing placeholder {e} in template: {template[:50]}...")
            return None
    
    def generate_mixed_sentences(self) -> List[Dict]:
        """Generate sentences with mixed names and numbers"""
        sentences = []
        samples_per_name = self.config['generation']['mixed']['samples_per_name']
        
        for person in self.all_names:
            for _ in range(samples_per_name):
                # Random template
                template = random.choice(self.mixed_templates)
                
                # Generate all number placeholders
                placeholders = self.generate_all_placeholders()
                
                # Fill template
                sentence = self.fill_template(template, person, placeholders)
                
                if sentence and len(sentence) >= self.config['generation']['mixed']['min_sentence_length']:
                    sentences.append({
                        'text': sentence,
                        'type': 'mixed',
                        'person': person['name'],
                        'template_preview': template[:50]
                    })
        
        return sentences
    
    def generate_pure_name_sentences(self) -> List[Dict]:
        """Generate pure name sentences (optional variety)"""
        sentences = []
        samples_per_name = self.config['generation']['names']['samples_per_name']
        
        for person in self.all_names:
            for _ in range(samples_per_name):
                template = random.choice(self.name_templates)
                
                values = {
                    'name': person.get('name', ''),
                    'honorific': person.get('honorific', ''),
                    'position': person.get('position', person.get('title', '')),
                    'party': person.get('party', ''),
                }
                
                try:
                    sentence = template.format(**values)
                    if len(sentence) >= self.config['generation']['names']['min_sentence_length']:
                        sentences.append({
                            'text': sentence,
                            'type': 'name',
                            'person': person['name']
                        })
                except KeyError:
                    continue
        
        return sentences
    
    def generate_pure_number_sentences(self) -> List[Dict]:
        """Generate pure number sentences (optional variety)"""
        sentences = []
        total_samples = self.config['generation']['numbers']['total_samples']
        
        for _ in range(total_samples):
            template = random.choice(self.number_templates)
            placeholders = self.generate_all_placeholders()
            
            try:
                sentence = template.format(**placeholders)
                if len(sentence) >= self.config['generation']['numbers']['min_sentence_length']:
                    sentences.append({
                        'text': sentence,
                        'type': 'number'
                    })
            except KeyError:
                continue
        
        return sentences
    
    def deduplicate_sentences(self, sentences: List[Dict]) -> List[Dict]:
        """Remove duplicate sentences based on text content"""
        seen_texts = set()
        unique_sentences = []
        duplicates = 0
        
        for sent in sentences:
            text = sent['text'].strip().lower()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_sentences.append(sent)
            else:
                duplicates += 1
        
        if duplicates > 0:
            logger.warning(f"Removed {duplicates:,} duplicate sentences")
        
        return unique_sentences
    
    def load_existing_sentences(self, existing_file: str) -> set:
        """Load existing sentences to avoid duplicates in incremental generation"""
        if not Path(existing_file).exists():
            return set()
        
        try:
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            
            existing_texts = {s['text'].strip().lower() for s in existing}
            logger.info(f"Loaded {len(existing_texts):,} existing sentences to avoid duplicates")
            return existing_texts
        except Exception as e:
            logger.warning(f"Could not load existing sentences: {e}")
            return set()
    
    def generate_all(self, output_file: str, existing_file: str = None, max_attempts: int = 3, max_sentences: int = None):
        """Generate all sentences and save to file"""
        logger.info("="*70)
        logger.info("Starting sentence generation")
        if max_sentences:
            logger.info(f"Target: {max_sentences} sentences (≈{max_sentences * 10 / 60:.1f} minutes of audio)")
        logger.info("="*70)
        
        # Load existing sentences to avoid duplicates
        existing_texts = set()
        if existing_file:
            existing_texts = self.load_existing_sentences(existing_file)
        
        all_sentences = []
        seen_texts = set()  # Track texts in current batch to avoid duplicates
        
        # Generate mixed sentences (priority)
        if self.config['generation'].get('use_mixed_templates', True):
            logger.info("\n📝 Generating mixed sentences (names + numbers)...")
            
            # If max_sentences specified, keep generating until we reach target
            if max_sentences:
                generation_round = 0
                max_rounds = 100  # Safety limit to prevent infinite loops
                
                while len(all_sentences) < max_sentences and generation_round < max_rounds:
                    generation_round += 1
                    logger.info(f"Generation round {generation_round}: {len(all_sentences)}/{max_sentences} sentences")
                    
                    mixed = self.generate_mixed_sentences()
                    
                    # Filter out duplicates with existing and already generated
                    new_sentences = []
                    for s in mixed:
                        text_lower = s['text'].strip().lower()
                        if text_lower not in existing_texts and text_lower not in seen_texts:
                            new_sentences.append(s)
                            seen_texts.add(text_lower)
                    
                    all_sentences.extend(new_sentences)
                    logger.info(f"  Added {len(new_sentences):,} unique sentences (total: {len(all_sentences):,})")
                    
                    # Stop if we've reached target
                    if len(all_sentences) >= max_sentences:
                        break
                    
                    # If we're not making progress, warn and break
                    if len(new_sentences) < 10:
                        logger.warning(f"Low unique sentences generated ({len(new_sentences)}), may have exhausted templates")
                        logger.warning(f"Consider generating in smaller batches or using different seeds")
                        break
            else:
                # Original behavior: generate based on config
                for attempt in range(max_attempts):
                    mixed = self.generate_mixed_sentences()
                    
                    # Filter out duplicates with existing
                    if existing_texts:
                        before = len(mixed)
                        mixed = [s for s in mixed if s['text'].strip().lower() not in existing_texts]
                        filtered = before - len(mixed)
                        if filtered > 0:
                            logger.info(f"Filtered out {filtered:,} duplicates with existing sentences")
                    
                    all_sentences.extend(mixed)
                    
                    if len(all_sentences) >= self.config['generation']['mixed']['samples_per_name'] * len(self.all_names):
                        break
                    
                    if attempt < max_attempts - 1:
                        logger.info(f"Generating more sentences (attempt {attempt + 2}/{max_attempts})...")
            
            logger.info(f"Generated {len(all_sentences):,} mixed sentences")
        
        # Generate pure name sentences (optional)
        if self.config['generation']['names']['samples_per_name'] > 0:
            logger.info("\n📝 Generating pure name sentences...")
            names = self.generate_pure_name_sentences()
            
            # Filter out duplicates with existing
            if existing_texts:
                before = len(names)
                names = [s for s in names if s['text'].strip().lower() not in existing_texts]
                filtered = before - len(names)
                if filtered > 0:
                    logger.info(f"Filtered out {filtered:,} duplicates with existing sentences")
            
            all_sentences.extend(names)
            logger.info(f"Generated {len(names):,} name sentences")
        
        # Generate pure number sentences (optional)
        if self.config['generation']['numbers']['total_samples'] > 0:
            logger.info("\n📝 Generating pure number sentences...")
            numbers = self.generate_pure_number_sentences()
            
            # Filter out duplicates with existing
            if existing_texts:
                before = len(numbers)
                numbers = [s for s in numbers if s['text'].strip().lower() not in existing_texts]
                filtered = before - len(numbers)
                if filtered > 0:
                    logger.info(f"Filtered out {filtered:,} duplicates with existing sentences")
            
            all_sentences.extend(numbers)
            logger.info(f"Generated {len(numbers):,} number sentences")
        
        # Deduplicate within this batch (only if not already done in generation loop)
        if not max_sentences or len(all_sentences) > max_sentences:
            logger.info("\n🔍 Removing duplicates within batch...")
            all_sentences = self.deduplicate_sentences(all_sentences)
        
        # Shuffle for variety
        random.shuffle(all_sentences)
        
        # Limit to max_sentences if we somehow generated more
        if max_sentences and len(all_sentences) > max_sentences:
            logger.info(f"\n✂️  Limiting output to {max_sentences} sentences (from {len(all_sentences):,} generated)")
            all_sentences = all_sentences[:max_sentences]
        
        # If we didn't reach target, warn user
        if max_sentences and len(all_sentences) < max_sentences:
            logger.warning(f"\n⚠️  Only generated {len(all_sentences):,} sentences (target was {max_sentences:,})")
            logger.warning(f"  Template combinations may be exhausted. Consider:")
            logger.warning(f"  - Adding more templates to sentence_templates/*.json")
            logger.warning(f"  - Adding more names to data_sources/*.json")
            logger.warning(f"  - Generating in smaller batches with different seeds")
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, ensure_ascii=False, indent=2)
        
        # Statistics
        logger.info("\n" + "="*70)
        logger.info("✅ Generation completed!")
        logger.info("="*70)
        logger.info(f"Total sentences: {len(all_sentences):,}")
        logger.info(f"Output file: {output_path}")
        
        # Breakdown by type
        type_counts = {}
        for sent in all_sentences:
            sent_type = sent.get('type', 'unknown')
            type_counts[sent_type] = type_counts.get(sent_type, 0) + 1
        
        logger.info("\nBreakdown by type:")
        for sent_type, count in type_counts.items():
            logger.info(f"  {sent_type}: {count:,} ({count/len(all_sentences)*100:.1f}%)")
        
        # Text length statistics
        lengths = [len(s['text']) for s in all_sentences]
        logger.info(f"\nText length statistics:")
        logger.info(f"  Average: {sum(lengths)/len(lengths):.1f} characters")
        logger.info(f"  Min: {min(lengths)} characters")
        logger.info(f"  Max: {max(lengths)} characters")
        
        # Sample sentences
        logger.info(f"\nSample sentences:")
        for i, sent in enumerate(random.sample(all_sentences, min(5, len(all_sentences))), 1):
            logger.info(f"  {i}. {sent['text']}")
        
        logger.info("="*70)
        
        return all_sentences


def main():
    parser = argparse.ArgumentParser(
        description="Generate training sentences with Malaysian names and numbers"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Name for this generation run (creates outputs/<name>/ folder). If not specified, uses --output path.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for generated sentences (overrides --name if specified)'
    )
    parser.add_argument(
        '--existing',
        type=str,
        help='Path to existing sentences file to avoid duplicates (for incremental generation)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (use different seeds for different batches)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=3,
        help='Maximum attempts to generate unique sentences'
    )
    parser.add_argument(
        '--max-sentences',
        type=int,
        help='Maximum number of sentences to output (e.g., 60 for ~10 min, 120 for ~20 min)'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.name:
        # Create folder structure: outputs/<name>/
        output_dir = Path('outputs') / args.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'sentences.json'
        logger.info(f"Using run name: {args.name}")
        logger.info(f"Output directory: {output_dir}")
    elif args.output:
        output_file = Path(args.output)
    else:
        # Default
        output_file = Path('outputs/sentences.json')
    
    # Set random seed
    random.seed(args.seed)
    
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output file: {output_file}")
    if args.existing:
        logger.info(f"Checking for duplicates against: {args.existing}")
    
    # Generate sentences
    generator = SentenceGenerator(args.config)
    sentences = generator.generate_all(
        str(output_file),
        existing_file=args.existing,
        max_attempts=args.max_attempts,
        max_sentences=args.max_sentences
    )
    
    logger.info(f"\n✅ Generated {len(sentences):,} unique sentences successfully!")
    logger.info(f"📁 Saved to: {output_file}")


if __name__ == "__main__":
    main()

