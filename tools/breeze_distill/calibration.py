"""Phoneme-complete calibration text and ARPAbet coverage analyzer for Breeze-TTS-2.

Provides the standardized 39-ARPAbet pangram reference text and validation utilities
to ensure zero-shot voice design or cloning clips have adequate phonetic diversity.

NOTE (F8): this is WORD-LEVEL LEXICAL ANALYSIS, not true phonetic analysis.
Coverage = union of phonemes from a word lookup table (CMUdict when available,
otherwise a small hand lexicon). It does not model stress, sub-lexemes, or
coarticulation, so 39/39 is a necessary-but-not-sufficient diversity check.
"""
from dataclasses import dataclass, field
import os
import re
from typing import Dict, List, Optional, Set, Tuple

# Full standard 39 ARPAbet phoneme inventory (stress digits stripped).
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}
ARPABET_CONSONANTS = {
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
    "T", "TH", "V", "W", "Y", "Z", "ZH"
}
ALL_ARPABET_39 = frozenset(ARPABET_VOWELS | ARPABET_CONSONANTS)

# Standardized default calibration passage engineered to cover all 39 ARPAbet phonemes
# in a single, natural, unhurried 29-word phrase with punctuation and a vocal anchor.
DEFAULT_CALIBRATION_PASSAGE = (
    "(sigh) With quick joy, the glad young boy paused to watch great, peaceful measures "
    "of rich golden music echo through the woods; why should we doubt his clear vision?"
)

# Reference lexicon for calibration passages, standard voices, and common English words.
BASE_LEXICON: Dict[str, List[str]] = {
    # Default pangram words
    "sigh": ["S", "AY"],
    "with": ["W", "IH", "DH"],
    "quick": ["K", "W", "IH", "K"],
    "joy": ["JH", "OY"],
    "the": ["DH", "AH"],
    "glad": ["G", "L", "AE", "D"],
    "young": ["Y", "AH", "NG"],
    "boy": ["B", "OY"],
    "paused": ["P", "AO", "Z", "D"],
    "to": ["T", "UW"],
    "watch": ["W", "AA", "CH"],
    "great": ["G", "R", "EY", "T"],
    "peaceful": ["P", "IY", "S", "F", "AH", "L"],
    "measures": ["M", "EH", "ZH", "ER", "Z"],
    "measure": ["M", "EH", "ZH", "ER"],
    "of": ["AH", "V"],
    "rich": ["R", "IH", "CH"],
    "golden": ["G", "OW", "L", "D", "AH", "N"],
    "music": ["M", "Y", "UW", "Z", "IH", "K"],
    "echo": ["EH", "K", "OW"],
    "through": ["TH", "R", "UW"],
    "woods": ["W", "UH", "D", "Z"],
    "why": ["W", "AY"],
    "should": ["SH", "UH", "D"],
    "we": ["W", "IY"],
    "doubt": ["D", "AW", "T"],
    "his": ["HH", "IH", "Z"],
    "clear": ["K", "L", "IH", "R"],
    "vision": ["V", "IH", "ZH", "AH", "N"],
    # Steward reference line words
    "good": ["G", "UH", "D"],
    "evening": ["IY", "V", "N", "IH", "NG"],
    "household": ["HH", "AW", "S", "HH", "OW", "L", "D"],
    "is": ["IH", "Z"],
    "in": ["IH", "N"],
    "order": ["AO", "R", "D", "ER"],
    "and": ["AE", "N", "D"],
    "i": ["AY"],
    "have": ["HH", "AE", "V"],
    "week": ["W", "IY", "K"],
    "weeks": ["W", "IY", "K", "S"],
    "list": ["L", "IH", "S", "T"],
    "ready": ["R", "EH", "D", "IY"],
    "whenever": ["W", "EH", "N", "EH", "V", "ER"],
    "you": ["Y", "UW"],
    "wish": ["W", "IH", "SH"],
    "there": ["DH", "EH", "R"],
    "no": ["N", "OW"],
    "hurry": ["HH", "ER", "IY"],
    "at": ["AE", "T"],
    "all": ["AO", "L"],
    "plenty": ["P", "L", "EH", "N", "T", "IY"],
    "time": ["T", "AY", "M"],
    # Calliope reference line words
    "not": ["N", "AA", "T"],
    "even": ["IY", "V", "AH", "N"],
    "fires": ["F", "AY", "ER", "Z"],
    "fire": ["F", "AY", "ER"],
    "hell": ["HH", "EH", "L"],
    "are": ["AA", "R"],
    "a": ["AH"],
    "match": ["M", "AE", "CH"],
    "for": ["F", "AO", "R"],
    "my": ["M", "AY"],
    "flame": ["F", "L", "EY", "M"],
    "flames": ["F", "L", "EY", "M", "Z"],
    "this": ["DH", "IH", "S"],
    "ill": ["AY", "L"],
    "start": ["S", "T", "AA", "R", "T"],
    "they": ["DH", "EY"],
    "can": ["K", "AE", "N"],
    "never": ["N", "EH", "V", "ER"],
    "put": ["P", "UH", "T"],
    "out": ["AW", "T"],
    "somebodys": ["S", "AH", "M", "B", "AA", "D", "IY", "Z"],
    "somebody": ["S", "AH", "M", "B", "AA", "D", "IY"],
    "dreams": ["D", "R", "IY", "M", "Z"],
    "just": ["JH", "AH", "S", "T"],
    "went": ["W", "EH", "N", "T"],
    "up": ["AH", "P"],
    "better": ["B", "EH", "T", "ER"],
    "break": ["B", "R", "EY", "K"],
    "extinguisher": ["IH", "K", "S", "T", "IH", "NG", "G", "W", "IH", "SH", "ER"],
}


@dataclass
class PhonemeReport:
    """Detailed phoneme coverage report for a calibration script."""
    text: str
    total_target_phonemes: int
    covered_phonemes_count: int
    coverage_ratio: float
    covered_phonemes: Set[str] = field(default_factory=set)
    missing_phonemes: Set[str] = field(default_factory=set)
    vowels_covered: Set[str] = field(default_factory=set)
    vowels_missing: Set[str] = field(default_factory=set)
    consonants_covered: Set[str] = field(default_factory=set)
    consonants_missing: Set[str] = field(default_factory=set)
    unrecognized_words: List[str] = field(default_factory=list)
    vocal_events: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True if all 39 ARPAbet phonemes are covered."""
        return len(self.missing_phonemes) == 0

    def summary(self) -> str:
        status = "COMPLETE (39/39)" if self.is_complete else f"PARTIAL ({self.covered_phonemes_count}/39)"
        pct = self.coverage_ratio * 100.0
        lines = [
            f"Phoneme Coverage Scorecard: {status} [{pct:.1f}%]",
            f"  - Target inventory: {self.total_target_phonemes} ARPAbet units",
            f"  - Covered ({len(self.covered_phonemes)}): {', '.join(sorted(self.covered_phonemes))}",
        ]
        if self.missing_phonemes:
            lines.append(f"  - Missing ({len(self.missing_phonemes)}): {', '.join(sorted(self.missing_phonemes))}")
            if self.vowels_missing:
                lines.append(f"    Missing vowels: {', '.join(sorted(self.vowels_missing))}")
            if self.consonants_missing:
                lines.append(f"    Missing consonants: {', '.join(sorted(self.consonants_missing))}")
        if self.vocal_events:
            lines.append(f"  - Vocal event anchors: {', '.join(self.vocal_events)}")
        if self.unrecognized_words:
            lines.append(f"  - Unrecognized words (skipped): {', '.join(self.unrecognized_words)}")
        return "\n".join(lines)


def extract_vocal_events(text: str) -> Tuple[str, List[str]]:
    """Extract parenthesized vocal event tags (e.g. (sigh), (clears throat))."""
    events = re.findall(r"\(([a-zA-Z\s]+)\)", text)
    cleaned = re.sub(r"\([a-zA-Z\s]+\)", "", text)
    return cleaned, events


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    cleaned, _ = extract_vocal_events(text)
    # Strip non-alphanumeric except apostrophes
    words = re.findall(r"[a-zA-Z']+", cleaned.lower())
    # Strip leading/trailing apostrophes
    return [w.strip("'") for w in words if w.strip("'")]


# CMUdict (e.g. /usr/share/dict/cmu dict, cmudict-0.7b, or any CMUdict flat file):
# one "WORD  PH1 PH2 ..." entry per line. Preferred over the hand lexicon (F8).
_CACHED_CMU: Dict[str, List[str]] = {}


def load_cmudict(path: Optional[str] = None) -> Dict[str, List[str]]:
    if path is None:
        path = os.environ.get("CMUDICT_PATH", "cmudict-0.7b")
    if path in _CACHED_CMU:
        return _CACHED_CMU[path]
    lex: Dict[str, List[str]] = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0].split("-")[0].lower()
                    # Strip stress digits (1/2/3) from phoneme tokens
                    lex[word] = [re.sub(r"[123]", "", p) for p in parts[1:]]
    except OSError:
        return {}
    _CACHED_CMU[path] = lex
    return lex


def analyze_phoneme_coverage(
    text: str, custom_lexicon: Optional[Dict[str, List[str]]] = None
) -> PhonemeReport:
    """Analyze the ARPAbet 39 coverage of a given reference text (lexical, see module note)."""
    lexicon = dict(BASE_LEXICON)
    cmu = load_cmudict()
    lexicon.update(cmu)
    if custom_lexicon:
        lexicon.update(custom_lexicon)

    cleaned_text, vocal_events = extract_vocal_events(text)
    tokens = tokenize_words(cleaned_text)

    covered: Set[str] = set()
    unrecognized: List[str] = []

    for word in tokens:
        lookup_word = word.replace("'", "")
        if word in lexicon:
            for p in lexicon[word]:
                if p in ALL_ARPABET_39:
                    covered.add(p)
        elif lookup_word in lexicon:
            for p in lexicon[lookup_word]:
                if p in ALL_ARPABET_39:
                    covered.add(p)
        else:
            # Fallback heuristic for common suffixes
            matched = False
            if lookup_word.endswith("s") and lookup_word[:-1] in lexicon:
                for p in lexicon[lookup_word[:-1]]:
                    if p in ALL_ARPABET_39:
                        covered.add(p)
                covered.add("Z")
                matched = True
            elif lookup_word.endswith("ed") and lookup_word[:-2] in lexicon:
                for p in lexicon[lookup_word[:-2]]:
                    if p in ALL_ARPABET_39:
                        covered.add(p)
                covered.add("D")
                matched = True
            if not matched:
                unrecognized.append(word)

    missing = set(ALL_ARPABET_39 - covered)
    vowels_cov = covered & ARPABET_VOWELS
    vowels_mis = missing & ARPABET_VOWELS
    cons_cov = covered & ARPABET_CONSONANTS
    cons_mis = missing & ARPABET_CONSONANTS

    ratio = len(covered) / len(ALL_ARPABET_39) if ALL_ARPABET_39 else 0.0

    return PhonemeReport(
        text=text,
        total_target_phonemes=len(ALL_ARPABET_39),
        covered_phonemes_count=len(covered),
        coverage_ratio=ratio,
        covered_phonemes=covered,
        missing_phonemes=missing,
        vowels_covered=vowels_cov,
        vowels_missing=vowels_mis,
        consonants_covered=cons_cov,
        consonants_missing=cons_mis,
        unrecognized_words=unrecognized,
        vocal_events=vocal_events,
    )


if __name__ == "__main__":
    import sys
    test_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CALIBRATION_PASSAGE
    print(f"Analyzing text:\n\"{test_text}\"\n")
    report = analyze_phoneme_coverage(test_text)
    print(report.summary())
