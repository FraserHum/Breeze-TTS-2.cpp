"""breeze_distill: The Vulkan Voice Foundry and Deployment Compiler for Breeze-TTS-2."""

from .calibration import (
    DEFAULT_CALIBRATION_PASSAGE,
    ALL_ARPABET_39,
    analyze_phoneme_coverage,
    extract_vocal_events,
    tokenize_words,
)
from .audition import (
    DEFAULT_SEEDS,
    generate_audition_reel,
    select_audition_candidate,
    analyze_wav_file,
)
from .pipeline import (
    DistillPipeline,
    DistillPipelineConfig,
    HARDWARE_PROFILES,
    VoiceSpec,
    parse_voice_flag,
)

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_CALIBRATION_PASSAGE",
    "ALL_ARPABET_39",
    "analyze_phoneme_coverage",
    "extract_vocal_events",
    "tokenize_words",
    "DEFAULT_SEEDS",
    "generate_audition_reel",
    "select_audition_candidate",
    "analyze_wav_file",
    "DistillPipeline",
    "DistillPipelineConfig",
    "HARDWARE_PROFILES",
    "VoiceSpec",
    "parse_voice_flag",
]
