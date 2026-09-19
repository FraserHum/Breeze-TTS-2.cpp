"""Unit and integration tests for breeze_distill."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.breeze_distill.calibration import (
    DEFAULT_CALIBRATION_PASSAGE,
    ALL_ARPABET_39,
    analyze_phoneme_coverage,
    extract_vocal_events,
    tokenize_words,
)
from tools.breeze_distill.audition import (
    generate_audition_reel,
    select_audition_candidate,
)
from tools.breeze_distill.pipeline import (
    DistillPipeline,
    DistillPipelineConfig,
    HARDWARE_PROFILES,
    REF_BUDGET_SAMPLES,
    REF_GAP_SAMPLES,
    VoiceSpec,
    plan_reference_concat,
    read_pcm_wav,
    write_pcm_wav,
)
from tools.breeze_distill import parse_voice_flag
import struct
import wave


class TestCalibration(unittest.TestCase):
    def test_arpabet_inventory_size(self):
        self.assertEqual(len(ALL_ARPABET_39), 39)

    def test_default_pangram_completeness(self):
        report = analyze_phoneme_coverage(DEFAULT_CALIBRATION_PASSAGE)
        self.assertTrue(report.is_complete, f"Default passage is missing: {report.missing_phonemes}")
        self.assertEqual(report.covered_phonemes_count, 39)
        self.assertEqual(len(report.missing_phonemes), 0)
        self.assertAlmostEqual(report.coverage_ratio, 1.0)
        self.assertIn("sigh", report.vocal_events)

    def test_partial_text_coverage(self):
        short_text = "Good evening. Have plenty of time."
        report = analyze_phoneme_coverage(short_text)
        self.assertFalse(report.is_complete)
        self.assertGreater(report.covered_phonemes_count, 10)
        self.assertLess(report.covered_phonemes_count, 39)
        self.assertGreater(len(report.missing_phonemes), 0)

    def test_vocal_event_extraction(self):
        text = "(sigh) Hello there, (clears throat) welcome."
        cleaned, events = extract_vocal_events(text)
        self.assertEqual(events, ["sigh", "clears throat"])
        self.assertNotIn("sigh", cleaned)
        self.assertNotIn("clears throat", cleaned)

    def test_tokenize_words(self):
        text = "With quick joy, the boy's music!"
        tokens = tokenize_words(text)
        self.assertIn("with", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("joy", tokens)
        self.assertIn("the", tokens)
        self.assertIn("boy's", tokens)
        self.assertIn("music", tokens)


class TestAudition(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audition_reel_generation_dry_run(self):
        out_dir = os.path.join(self.temp_dir, "aud_test")
        manifest = generate_audition_reel(
            instruction="A calm and warm Butler",
            voice_name="steward_test",
            out_dir=out_dir,
            seeds=[42, 77],
            dry_run=True,
        )

        self.assertEqual(manifest["voice_name"], "steward_test")
        self.assertEqual(len(manifest["candidates"]), 2)
        self.assertEqual(manifest["candidates"][0]["seed"], 42)
        self.assertEqual(manifest["candidates"][1]["seed"], 77)

        manifest_path = Path(out_dir) / "audition_reel.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path) as f:
            data = json.load(f)
            self.assertEqual(data["voice_name"], "steward_test")

    def test_audition_selection_dry_run(self):
        aud_dir = os.path.join(self.temp_dir, "aud_test")
        voices_dir = os.path.join(self.temp_dir, "voices")

        # Generate mock reel first
        generate_audition_reel(
            instruction="A calm and warm Butler",
            voice_name="steward_test",
            out_dir=aud_dir,
            seeds=[42],
            dry_run=True,
        )

        res = select_audition_candidate(
            voice_name="steward_test",
            chosen_seed=42,
            audition_dir=aud_dir,
            voices_dir=voices_dir,
            dry_run=True,
        )
        self.assertEqual(res["voice_name"], "steward_test")
        self.assertEqual(res["chosen_seed"], 42)
        self.assertEqual(res["status"], "dry_run_selected")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_voice_flag(self):
        v1 = parse_voice_flag("name=calliope,type=clone,audio=calliope.wav")
        self.assertEqual(v1.name, "calliope")
        self.assertEqual(v1.voice_type, "clone")
        self.assertEqual(v1.audio_path, "calliope.wav")
        self.assertEqual(v1.utterances, [("calliope.wav", None)])

        v2 = parse_voice_flag('name=steward,type=design,instruction="Warm butler",seed=108')
        self.assertEqual(v2.name, "steward")
        self.assertEqual(v2.voice_type, "design")
        self.assertEqual(v2.instruction, "Warm butler")
        self.assertEqual(v2.seed, 108)

    def test_parse_voice_flag_multi_utterance(self):
        # F8: repeated audio=/text= pairs become multi-utterance clone references;
        # quoted commas in text must stay inside the value.
        v = parse_voice_flag(
            'name=calliope,type=clone,audio=a1.wav,text="line, one",audio=a2.wav,text="line two"'
        )
        self.assertEqual(v.utterances, [("a1.wav", "line, one"), ("a2.wav", "line two")])
        self.assertEqual(v.audio_path, "a1.wav")
        self.assertEqual(v.transcript, "line, one")
        v.validate()  # structural: 2 utterances is valid shape

    def test_voice_spec_validation(self):
        bad_v = VoiceSpec(name="bad name with spaces", voice_type="clone")
        with self.assertRaises(ValueError):
            bad_v.validate()

        bad_type = VoiceSpec(name="voice", voice_type="unknown")
        with self.assertRaises(ValueError):
            bad_type.validate()

    def test_hardware_profiles(self):
        # Receipt-gated profiles only; the unmeasured 6-block 'edge' profile was CUT.
        self.assertIn("780m", HARDWARE_PROFILES)
        self.assertEqual(HARDWARE_PROFILES["780m"]["n_blocks"], 9)
        self.assertIn("baseline", HARDWARE_PROFILES)
        self.assertEqual(HARDWARE_PROFILES["baseline"]["n_blocks"], 12)
        self.assertNotIn("edge", HARDWARE_PROFILES)

    def test_pipeline_dry_run_execution(self):
        work_dir = os.path.join(self.temp_dir, "distill_work")
        out_model = os.path.join(self.temp_dir, "out_q4_k.gguf")

        voices = [
            VoiceSpec(name="calliope", voice_type="clone", audio_path="tests/calliope.wav"),
            VoiceSpec(name="steward", voice_type="design", instruction="Warm butler", seed=42),
        ]

        cfg = DistillPipelineConfig(
            voices=voices,
            target_hardware="780m",
            work_dir=work_dir,
            output_model=out_model,
            dry_run=True,
        )

        pipeline = DistillPipeline(cfg)
        final_model = pipeline.run()

        self.assertEqual(str(final_model), out_model)
        self.assertTrue(Path(out_model).exists())

        scorecard_path = Path(work_dir) / "scorecard.json"
        self.assertTrue(scorecard_path.exists())
        with open(scorecard_path) as f:
            card = json.load(f)
            self.assertEqual(card["n_blocks"], 9)
            self.assertEqual(card["target_hardware"], "780m")
            self.assertEqual(card["voices"], ["calliope", "steward"])
            # F4/F5: dry-run has no threshold => fail-closed UNVERIFIED,
            # all_pass is None (never fabricated True), no RTF number.
            self.assertIsNone(card["all_pass"])
            self.assertEqual(card["safety_gate"]["status"], "unverified")
            self.assertIsNone(card["artifact_rtf"])

            # F8: no real synthesis ran in dry-run => no hold-out number is fabricated.
            self.assertIsNone(card["clone_holdout"])


class TestCalliopeMethod(unittest.TestCase):
    """F8 Calliope Method: concat references into the receipted 161-token budget."""

    RATE = 24000

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _wav(self, name, seconds):
        path = Path(self.temp_dir) / name
        n = int(seconds * self.RATE)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(struct.pack(f"<{n}h", *([1000] * n)))
        return str(path)

    def test_plan_fits_no_holdout(self):
        a = self._wav("a.wav", 6.0)
        b = self._wav("b.wav", 6.0)
        included, holdout = plan_reference_concat([(a, "x"), (b, "y")])
        self.assertEqual(included, [0, 1])
        self.assertIsNone(holdout)

    def test_plan_overflow_first_nonfitting_is_holdout(self):
        a = self._wav("a.wav", 8.0)
        b = self._wav("b.wav", 5.0)  # 8 + 0.25 + 5 = 13.25 s > 12.88 s budget
        c = self._wav("c.wav", 0.5)
        included, holdout = plan_reference_concat([(a, "x"), (b, "y"), (c, "z")])
        self.assertEqual(holdout, 1)
        self.assertIn(2, included)  # later smaller utterance still packs
        self.assertEqual(included.count(1), 0)
        # budget accounting: 8.0 + (0.25 + 0.5) s
        self.assertEqual(included, [0, 2])

    def test_wav_io_roundtrip_and_fail_closed(self):
        a = self._wav("a.wav", 1.0)
        pcm = read_pcm_wav(a)
        self.assertEqual(len(pcm), self.RATE)
        write_pcm_wav(Path(self.temp_dir) / "b.wav", pcm)
        self.assertEqual(read_pcm_wav(str(Path(self.temp_dir) / "b.wav")), pcm)
        # wrong format must fail closed (Calliope receipt: 24 kHz s16 mono)
        bad = Path(self.temp_dir) / "bad.wav"
        with wave.open(str(bad), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<48000h", *([1] * 48000)))
        with self.assertRaises(ValueError):
            read_pcm_wav(str(bad))

    def test_scorecard_never_fabricates_unmeasured_metrics(self):
        # F4/F5 extension: dry-run telemetry is absent => Loss/KL/Top-1 print as
        # "Unmeasured", and a threshold cannot PASS a missing measurement.
        work_dir = os.path.join(self.temp_dir, "ws")
        cfg = DistillPipelineConfig(
            voices=[VoiceSpec(name="calliope", voice_type="clone",
                               audio_path="a.wav", utterances=[("a.wav", "x"), ("b.wav", "y")])],
            target_hardware="780m",
            work_dir=work_dir,
            output_model=os.path.join(self.temp_dir, "o.gguf"),
            dry_run=True,
            min_top1_threshold=85.0,
        )
        p = DistillPipeline(cfg)
        p.run()
        with open(Path(work_dir) / "scorecard.json") as f:
            card = json.load(f)
            # With a threshold but no telemetry, the gate must FAIL, not PASS (fail-closed).
            self.assertIsNone(card["safety_gate"]["measured"]["calliope"])
            self.assertEqual(card["safety_gate"]["status"], "failed")

    def test_holdout_analyzer_recovers_shift(self):
        # Known-shift recovery: synth == teacher shifted 12 samples (analyzer
        # machinery receipted in analyze_audio.py self_check; here we verify the
        # pipeline wrapper reports it).
        import subprocess
        import sys
        from tools.breeze_distill.pipeline import analyze_holdout

        t = Path(self.temp_dir) / "t.wav"
        n = self.RATE  # 1 s
        samples = [1000 + ((i * 7) % 171) - 85 for i in range(n)]
        write_pcm_wav(t, samples)
        write_pcm_wav(Path(self.temp_dir) / "s.wav", [0] * 12 + samples[: n - 12])
        m = analyze_holdout(str(t), str(Path(self.temp_dir) / "s.wav"), self.temp_dir)
        self.assertGreater(m["best_lag_samples"], 0)
        self.assertAlmostEqual(m["best_lag_samples"], 12, delta=1)
        # best_lag_correlation is the rescued overlap metric; raw correlation
        # is unaligned and not a shift-robust claim.
        self.assertGreater(m["best_lag_correlation"], 0.99)
        self.assertGreater(m["rms_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
