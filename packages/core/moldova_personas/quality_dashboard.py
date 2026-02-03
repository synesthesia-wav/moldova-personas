"""
Quality dashboard for dataset-level metrics.

Fast, deterministic monitoring of:
- Style/variety (opening moves, lexical diversity)
- Realism (constraint categories, consequence hit-rate)
- Anchors (top-50 shares, rare anchor compliance, region mutual info)
- Uniqueness (near-duplicate detection)
"""

import re
import math
from typing import Dict, List, Tuple, Any, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib


@dataclass
class DatasetQualityMetrics:
    """Complete quality metrics for a persona dataset."""
    
    # Metadata
    total_records: int
    timestamp: str
    
    # Style / Variety
    opening_move_distribution: Dict[str, float]
    trigram_repetition_rate: float
    lexical_diversity_per_field: Dict[str, float]
    sentence_starter_variety: Dict[str, float]
    
    # Realism
    constraint_category_distribution: Dict[str, float]
    constraint_consequence_hit_rate: float
    constraint_by_region: Dict[str, Dict[str, float]]
    
    # Anchors
    top_50_anchor_shares: List[Tuple[str, float]]
    rare_anchor_compliance_rate: float
    anchor_region_mutual_info: float
    anchor_overuse_violations: List[Tuple[str, float]]
    
    # Uniqueness
    duplicate_pairs_detected: int
    estimated_uniqueness_ratio: float
    minhash_similarity_distribution: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "timestamp": self.timestamp,
            "style_variety": {
                "opening_move_distribution": self.opening_move_distribution,
                "trigram_repetition_rate": self.trigram_repetition_rate,
                "lexical_diversity_per_field": self.lexical_diversity_per_field,
            },
            "realism": {
                "constraint_category_distribution": self.constraint_category_distribution,
                "constraint_consequence_hit_rate": self.constraint_consequence_hit_rate,
                "constraint_by_region": self.constraint_by_region,
            },
            "anchors": {
                "top_50_anchor_shares": self.top_50_anchor_shares[:10],  # Top 10 for brevity
                "rare_anchor_compliance_rate": self.rare_anchor_compliance_rate,
                "anchor_region_mutual_info": self.anchor_region_mutual_info,
                "overuse_violations": self.anchor_overuse_violations,
            },
            "uniqueness": {
                "duplicate_pairs": self.duplicate_pairs_detected,
                "uniqueness_ratio": self.estimated_uniqueness_ratio,
            }
        }


class QualityDashboard:
    """
    Dataset-level quality monitoring dashboard.
    
    Usage:
        dashboard = QualityDashboard()
        metrics = dashboard.analyze_dataset(personas)
        dashboard.print_report(metrics)
    """
    
    def __init__(
        self,
        max_anchor_share: float = 0.05,  # 5% cap
        min_rare_anchor_rate: float = 0.5,  # 50% must have rare anchor
        trigram_threshold: float = 0.1,  # Max 10% trigram repetition
        dedup_threshold: float = 0.8  # MinHash similarity threshold
    ):
        self.max_anchor_share = max_anchor_share
        self.min_rare_anchor_rate = min_rare_anchor_rate
        self.trigram_threshold = trigram_threshold
        self.dedup_threshold = dedup_threshold

        self._narrative_fields = [
            "persona",
            "professional_persona",
            "sports_persona",
            "arts_persona",
            "travel_persona",
            "culinary_persona",
        ]
        self._narrative_field_map = {
            "persona": "descriere_generala",
            "professional_persona": "profil_profesional",
            "sports_persona": "hobby_sport",
            "arts_persona": "hobby_arta_cultura",
            "travel_persona": "hobby_calatorii",
            "culinary_persona": "hobby_culinar",
        }

    def _get_narrative_text(self, persona: Dict, field: str) -> str:
        personas_block = persona.get("personas")
        if isinstance(personas_block, dict):
            text = personas_block.get(field, "")
            if isinstance(text, str) and text.strip():
                return text

        fallback_key = self._narrative_field_map.get(field)
        if fallback_key:
            text = persona.get(fallback_key, "")
            if isinstance(text, str):
                return text
        return ""

    def _get_all_narrative_texts(self, persona: Dict) -> List[str]:
        personas_block = persona.get("personas")
        if isinstance(personas_block, dict) and personas_block:
            return [v for v in personas_block.values() if isinstance(v, str)]

        texts = []
        for field in self._narrative_fields:
            text = self._get_narrative_text(persona, field)
            if text:
                texts.append(text)
        return texts
    
    def analyze_dataset(self, personas: List[Dict]) -> DatasetQualityMetrics:
        """Analyze full dataset and return quality metrics."""
        from datetime import datetime
        
        return DatasetQualityMetrics(
            total_records=len(personas),
            timestamp=datetime.now().isoformat(),
            opening_move_distribution=self._analyze_opening_moves(personas),
            trigram_repetition_rate=self._analyze_trigrams(personas),
            lexical_diversity_per_field=self._analyze_lexical_diversity(personas),
            sentence_starter_variety=self._analyze_sentence_starters(personas),
            constraint_category_distribution=self._analyze_constraint_categories(personas),
            constraint_consequence_hit_rate=self._analyze_constraint_consequences(personas),
            constraint_by_region=self._analyze_constraints_by_region(personas),
            top_50_anchor_shares=self._analyze_top_anchors(personas),
            rare_anchor_compliance_rate=self._analyze_rare_anchor_compliance(personas),
            anchor_region_mutual_info=self._calculate_anchor_region_mi(personas),
            anchor_overuse_violations=self._detect_anchor_overuse(personas),
            duplicate_pairs_detected=self._detect_duplicates(personas),
            estimated_uniqueness_ratio=self._estimate_uniqueness(personas),
            minhash_similarity_distribution=self._get_similarity_distribution(personas),
        )
    
    # ========== STYLE / VARIETY ==========
    
    def _analyze_opening_moves(self, personas: List[Dict]) -> Dict[str, float]:
        """Analyze distribution of opening rhetorical moves."""
        moves = Counter()
        total = 0
        
        opening_patterns = {
            "situational_hook": r"^(după|în|sâmbăta|duminica|la|când)",
            "preference_statement": r"^(preferă|îi place|evită|nu îi place|apreciază)",
            "micro_story": r"^(sâmbăta trecută|recent|acum câteva zile|vara trecută)",
            "contrast": r"^(deși|chiar dacă|cu toate că|totuși)",
            "value": r"^(ține|prețuiește|cont[eă]ază|îi pasă)",
            "sensory_anchor": r"^(mirosul|sunetul|gustul|imaginea|senzația)",
        }
        
        for p in personas:
            for field in self._narrative_fields:
                text = self._get_narrative_text(p, field).lower()
                if not text:
                    continue
                
                total += 1
                first_sentence = text.split('.')[0] if '.' in text else text[:50]
                
                for move, pattern in opening_patterns.items():
                    if re.search(pattern, first_sentence):
                        moves[move] += 1
                        break
                else:
                    moves["other"] += 1
        
        # Normalize to percentages
        return {move: count / max(total, 1) for move, count in moves.items()}
    
    def _analyze_trigrams(self, personas: List[Dict]) -> float:
        """Calculate trigram repetition rate."""
        trigram_counts = Counter()
        total_trigrams = 0
        
        for p in personas:
            for field in self._narrative_fields:
                text = self._get_narrative_text(p, field).lower()
                words = re.findall(r'\b\w+\b', text)
                
                for i in range(len(words) - 2):
                    trigram = ' '.join(words[i:i+3])
                    trigram_counts[trigram] += 1
                    total_trigrams += 1
        
        # Repetition rate = % of trigrams that appear more than once
        repeated = sum(1 for count in trigram_counts.values() if count > 1)
        return repeated / max(len(trigram_counts), 1)
    
    def _analyze_lexical_diversity(self, personas: List[Dict]) -> Dict[str, float]:
        """Calculate type-token ratio per field."""
        diversity = {}
        
        for field in self._narrative_fields:
            all_tokens = []
            
            for p in personas:
                text = self._get_narrative_text(p, field).lower()
                tokens = re.findall(r'\b\w+\b', text)
                all_tokens.extend(tokens)
            
            if all_tokens:
                unique_types = len(set(all_tokens))
                total_tokens = len(all_tokens)
                diversity[field] = unique_types / total_tokens
            else:
                diversity[field] = 0.0
        
        return diversity
    
    def _analyze_sentence_starters(self, personas: List[Dict]) -> Dict[str, float]:
        """Analyze variety in sentence starters."""
        starters = Counter()
        
        for p in personas:
            all_text = " ".join(self._get_all_narrative_texts(p))
            sentences = re.split(r'[.!?]+', all_text)
            
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    first_word = sent.split()[0].lower() if sent.split() else ""
                    if first_word:
                        starters[first_word] += 1
        
        # Calculate entropy of distribution
        total = sum(starters.values())
        if total == 0:
            return {"entropy": 0.0, "unique_ratio": 0.0}
        
        entropy = -sum((count/total) * math.log2(count/total) for count in starters.values())
        unique_ratio = len(starters) / total
        
        return {"entropy": entropy, "unique_ratio": unique_ratio}
    
    # ========== REALISM ==========
    
    def _analyze_constraint_categories(self, personas: List[Dict]) -> Dict[str, float]:
        """Analyze distribution of constraint categories."""
        categories = Counter()
        total = 0
        
        category_keywords = {
            "budget": ["buget", "bani", "economii", "ieftin", "scump"],
            "time_pressure": ["timp", "grabă", "ocupat", "program", "deadline"],
            "family_duty": ["familie", "copii", "nepoți", "părinți", "responsabil"],
            "health_nuisance": ["durere", "spate", "somn", "oboseală", "sănătate"],
            "commute_burden": ["drum", "navetă", "maxi-taxi", "autobuz", "trafic"],
            "bureaucratic_fatigue": ["hârtii", "birocrație", "primărie", "instituție"],
        }
        
        for p in personas:
            constraints = p.get("constraints", [])
            
            for constraint in constraints:
                constraint_lower = constraint.lower()
                matched = False
                
                for category, keywords in category_keywords.items():
                    if any(kw in constraint_lower for kw in keywords):
                        categories[category] += 1
                        total += 1
                        matched = True
                        break
                
                if not matched:
                    categories["other"] += 1
                    total += 1
        
        return {cat: count / max(total, 1) for cat, count in categories.items()}
    
    def _analyze_constraint_consequences(self, personas: List[Dict]) -> float:
        """Calculate how often constraints have visible consequences."""
        hit_count = 0
        total_constraints = 0
        
        consequence_keywords = {
            "budget": ["nu-și permite", "compară", "economisește", "ieftin"],
            "time": ["nu are timp", "grabă", "târziu", "dimineața devreme"],
            "family": ["copii", "nepoți", "părinți", "familie"],
            "health": ["durere", "repaus", "evită efort"],
        }
        
        for p in personas:
            constraints = p.get("constraints", [])
            persona_text = " ".join(self._get_all_narrative_texts(p)).lower()
            
            for constraint in constraints:
                total_constraints += 1
                
                # Check if any consequence keyword appears in persona text
                for category, keywords in consequence_keywords.items():
                    if any(kw in constraint.lower() for kw in keywords):
                        if any(kw in persona_text for kw in keywords):
                            hit_count += 1
                            break
        
        return hit_count / max(total_constraints, 1)
    
    def _analyze_constraints_by_region(self, personas: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Analyze constraint distribution by region."""
        by_region = defaultdict(Counter)
        
        category_keywords = {
            "budget": ["buget", "bani", "economii"],
            "time_pressure": ["timp", "grabă", "ocupat"],
            "family_duty": ["familie", "copii", "nepoți"],
        }
        
        for p in personas:
            region = p.get("region", "unknown")
            constraints = p.get("constraints", [])
            
            for constraint in constraints:
                for category, keywords in category_keywords.items():
                    if any(kw in constraint.lower() for kw in keywords):
                        by_region[region][category] += 1
        
        # Normalize
        return {
            region: {cat: count / sum(cats.values()) for cat, count in cats.items()}
            for region, cats in by_region.items()
        }
    
    # ========== ANCHORS ==========
    
    def _analyze_top_anchors(self, personas: List[Dict]) -> List[Tuple[str, float]]:
        """Get top 50 anchors by share."""
        anchor_counts = Counter()
        
        for p in personas:
            anchors = p.get("anchors", {})
            for anchor_type, anchor_value in anchors.items():
                if anchor_value:
                    anchor_counts[anchor_value] += 1
        
        total = len(personas)
        return [(anchor, count / total) for anchor, count in anchor_counts.most_common(50)]
    
    def _analyze_rare_anchor_compliance(self, personas: List[Dict]) -> float:
        """Calculate % of personas with at least one rare anchor."""
        # Define rare as appearing <1% of the time
        all_anchors = self._analyze_top_anchors(personas)
        frequent_anchors = {a for a, share in all_anchors if share > 0.01}
        
        rare_count = 0
        for p in personas:
            anchors = p.get("anchors", {}).values()
            if any(a not in frequent_anchors for a in anchors if a):
                rare_count += 1
        
        return rare_count / max(len(personas), 1)
    
    def _calculate_anchor_region_mi(self, personas: List[Dict]) -> float:
        """Calculate mutual information between anchors and regions."""
        # P(anchor, region)
        joint_counts = Counter()
        anchor_counts = Counter()
        region_counts = Counter()
        
        for p in personas:
            region = p.get("region", "unknown")
            anchors = p.get("anchors", {})
            
            region_counts[region] += 1
            
            for anchor_value in anchors.values():
                if anchor_value:
                    joint_counts[(anchor_value, region)] += 1
                    anchor_counts[anchor_value] += 1
        
        total = len(personas)
        
        # Calculate MI
        mi = 0.0
        for (anchor, region), joint_count in joint_counts.items():
            p_joint = joint_count / total
            p_anchor = anchor_counts[anchor] / total
            p_region = region_counts[region] / total
            
            if p_joint > 0 and p_anchor > 0 and p_region > 0:
                mi += p_joint * math.log2(p_joint / (p_anchor * p_region))
        
        return mi
    
    def _detect_anchor_overuse(self, personas: List[Dict]) -> List[Tuple[str, float]]:
        """Detect anchors exceeding max share threshold."""
        top_anchors = self._analyze_top_anchors(personas)
        return [(anchor, share) for anchor, share in top_anchors if share > self.max_anchor_share]
    
    # ========== UNIQUENESS ==========
    
    def _get_shingles(self, text: str, k: int = 3) -> Set[str]:
        """Get k-shingles (word n-grams) from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}
    
    def _minhash_signature(self, text: str, num_hashes: int = 20) -> List[int]:
        """Generate MinHash signature for text."""
        shingles = self._get_shingles(text, k=3)
        
        if not shingles:
            return [0] * num_hashes
        
        signature = []
        for i in range(num_hashes):
            # Simple hash function
            min_hash = min(
                int(hashlib.md5(f"{shingle}_{i}".encode()).hexdigest(), 16)
                for shingle in shingles
            )
            signature.append(min_hash)
        
        return signature
    
    def _estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def _detect_duplicates(self, personas: List[Dict]) -> int:
        """Detect near-duplicate pairs using MinHash."""
        # Sample for efficiency if large
        sample_size = min(len(personas), 1000)
        sample = personas[:sample_size]
        
        signatures = []
        for p in sample:
            text = " ".join(self._get_all_narrative_texts(p))
            sig = self._minhash_signature(text)
            signatures.append(sig)
        
        # Count pairs above threshold
        duplicate_pairs = 0
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                similarity = self._estimate_jaccard(signatures[i], signatures[j])
                if similarity > self.dedup_threshold:
                    duplicate_pairs += 1
        
        # Extrapolate to full dataset
        if len(personas) > sample_size:
            scale_factor = (len(personas) / sample_size) ** 2
            duplicate_pairs = int(duplicate_pairs * scale_factor)
        
        return duplicate_pairs
    
    def _estimate_uniqueness(self, personas: List[Dict]) -> float:
        """Estimate uniqueness ratio."""
        duplicates = self._detect_duplicates(personas)
        total_pairs = len(personas) * (len(personas) - 1) // 2
        
        if total_pairs == 0:
            return 1.0
        
        return 1.0 - (duplicates / total_pairs)
    
    def _get_similarity_distribution(self, personas: List[Dict]) -> List[float]:
        """Get distribution of pairwise similarities (sampled)."""
        sample_size = min(len(personas), 500)
        sample = personas[:sample_size]
        
        signatures = []
        for p in sample:
            text = " ".join(self._get_all_narrative_texts(p))
            sig = self._minhash_signature(text)
            signatures.append(sig)
        
        similarities = []
        for i in range(0, len(signatures), 10):  # Sample pairs
            for j in range(i + 1, min(i + 10, len(signatures))):
                sim = self._estimate_jaccard(signatures[i], signatures[j])
                similarities.append(sim)
        
        return sorted(similarities)
    
    # ========== REPORTING ==========
    
    def print_report(self, metrics: DatasetQualityMetrics):
        """Print formatted quality report."""
        print("\n" + "="*70)
        print("QUALITY DASHBOARD REPORT")
        print("="*70)
        print(f"Dataset size: {metrics.total_records:,} records")
        print(f"Timestamp: {metrics.timestamp}")
        
        print("\n📊 STYLE / VARIETY")
        print("-" * 40)
        print("Opening move distribution:")
        for move, share in sorted(metrics.opening_move_distribution.items(), key=lambda x: -x[1]):
            print(f"  {move}: {share:.1%}")
        print(f"\nTrigram repetition: {metrics.trigram_repetition_rate:.2%}")
        print(f"Lexical diversity (avg): {sum(metrics.lexical_diversity_per_field.values())/6:.3f}")
        
        print("\n✅ REALISM")
        print("-" * 40)
        print("Constraint categories:")
        for cat, share in sorted(metrics.constraint_category_distribution.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {share:.1%}")
        print(f"\nConstraint consequence hit rate: {metrics.constraint_consequence_hit_rate:.1%}")
        
        print("\n📍 ANCHORS")
        print("-" * 40)
        print("Top 10 anchors:")
        for anchor, share in metrics.top_50_anchor_shares[:10]:
            status = "⚠️ OVER" if share > self.max_anchor_share else "✓"
            print(f"  {status} {anchor}: {share:.1%}")
        print(f"\nRare anchor compliance: {metrics.rare_anchor_compliance_rate:.1%}")
        print(f"Anchor-region mutual info: {metrics.anchor_region_mutual_info:.3f}")
        
        if metrics.anchor_overuse_violations:
            print("\n⚠️  ANCHOR OVERUSE VIOLATIONS:")
            for anchor, share in metrics.anchor_overuse_violations:
                print(f"    {anchor}: {share:.1%} (max: {self.max_anchor_share:.1%})")
        
        print("\n🔍 UNIQUENESS")
        print("-" * 40)
        print(f"Duplicate pairs detected: {metrics.duplicate_pairs_detected}")
        print(f"Estimated uniqueness: {metrics.estimated_uniqueness_ratio:.1%}")
        
        # Overall assessment
        print("\n" + "="*70)
        issues = self._count_issues(metrics)
        if issues == 0:
            print("✅ OVERALL: PASSED all quality checks")
        elif issues < 3:
            print(f"⚠️  OVERALL: {issues} minor issues detected")
        else:
            print(f"❌ OVERALL: {issues} issues require attention")
        print("="*70 + "\n")
    
    def _count_issues(self, metrics: DatasetQualityMetrics) -> int:
        """Count number of quality issues."""
        issues = 0
        
        # Check opening distribution (should be roughly uniform)
        max_opening_share = max(metrics.opening_move_distribution.values())
        if max_opening_share > 0.4:  # No move should dominate >40%
            issues += 1
        
        # Check trigram repetition
        if metrics.trigram_repetition_rate > self.trigram_threshold:
            issues += 1
        
        # Check rare anchor compliance
        if metrics.rare_anchor_compliance_rate < self.min_rare_anchor_rate:
            issues += 1
        
        # Check anchor overuse
        if metrics.anchor_overuse_violations:
            issues += len(metrics.anchor_overuse_violations)
        
        # Check uniqueness
        if metrics.estimated_uniqueness_ratio < 0.9:
            issues += 1
        
        return issues


def quick_quality_check(personas: List[Dict]) -> Dict[str, Any]:
    """
    Quick quality check for smaller datasets.
    
    Returns pass/fail summary.
    """
    dashboard = QualityDashboard()
    metrics = dashboard.analyze_dataset(personas)
    
    issues = dashboard._count_issues(metrics)
    
    return {
        "total": len(personas),
        "issues": issues,
        "passed": issues == 0,
        "trigram_repetition": metrics.trigram_repetition_rate,
        "rare_anchor_compliance": metrics.rare_anchor_compliance_rate,
        "uniqueness": metrics.estimated_uniqueness_ratio,
    }
