"""End-to-end smoke test for the Transformers 5.19.0 Qwen3-TTS upgrade."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers

from qwen_tts import Qwen3TTSModel


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
EXPECTED_TRANSFORMERS = "5.19.0"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_TTS_MODEL", DEFAULT_MODEL),
        help=f"Hugging Face model id or local model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed as device_map to from_pretrained (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype (default: selected for the chosen device)",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation, such as sdpa",
    )
    parser.add_argument(
        "--text",
        default="Hello from Qwen3-TTS. This is a short compatibility smoke test.",
        help="Text to synthesize",
    )
    parser.add_argument("--language", default="English", help="Synthesis language")
    parser.add_argument("--speaker", default="Ryan", help="Custom voice speaker")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated audio tokens (default: 256)",
    )
    parser.add_argument(
        "--asr-model",
        default=os.environ.get("QWEN3_TTS_ASR_MODEL", "openai/whisper-tiny"),
        help="Speech recognition model used to check the spoken words",
    )
    parser.add_argument(
        "--asr-language",
        default="en",
        help="Whisper language code used for transcription (default: en)",
    )
    parser.add_argument(
        "--expected-text",
        default=None,
        help="Expected spoken text (defaults to --text)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable audio generation (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("smoke_test_output.wav"),
        help="Destination WAV path (default: smoke_test_output.wav)",
    )
    return parser.parse_args()


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = re.findall(r"[a-z0-9']+", expected.lower())
    actual_words = re.findall(r"[a-z0-9']+", actual.lower())
    if not expected_words:
        raise ValueError("Expected text must contain at least one word")

    previous = list(range(len(actual_words) + 1))
    for row, expected_word in enumerate(expected_words, start=1):
        current = [row]
        for column, actual_word in enumerate(actual_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected_words)


def main() -> None:
    args = parse_args()
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            f"Expected transformers=={EXPECTED_TRANSFORMERS}, "
            f"found {transformers.__version__}. Install this project's requirements first."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    print(f"Transformers: {transformers.__version__}")
    print(f"Loading {args.model} on {device} with {dtype}...")
    torch.manual_seed(args.seed)

    load_kwargs = {"device_map": device, "dtype": dtype}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3TTSModel.from_pretrained(args.model, **load_kwargs)

    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
    )
    if not wavs:
        raise RuntimeError("generate_custom_voice returned no audio")

    audio = np.asarray(wavs[0])
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("generate_custom_voice returned empty or non-finite audio")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise RuntimeError(f"Invalid sample rate returned: {sample_rate!r}")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, int(sample_rate))
    written_audio = sf.info(output)
    duration = written_audio.frames / written_audio.samplerate
    if (
        not math.isfinite(duration)
        or duration <= 0
        or written_audio.samplerate != int(sample_rate)
        or written_audio.frames == 0
    ):
        raise RuntimeError("Audio validation or WAV output failed")

    print(
        f"PASS: generated {duration:.2f}s of audio at {sample_rate} Hz; "
        f"saved {output}"
    )

    from transformers import pipeline

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=-1,
    )
    transcription = transcriber(
        {"raw": audio, "sampling_rate": int(sample_rate)},
        generate_kwargs={"language": args.asr_language, "task": "transcribe"},
    )["text"].strip()
    expected_text = args.expected_text or args.text
    error_rate = word_error_rate(expected_text, transcription)
    print(f"Expected speech: {expected_text}")
    print(f"Recognized speech: {transcription}")
    print(f"Word error rate: {error_rate:.0%}")
    if error_rate > 0.25:
        raise RuntimeError(
            "Audio transcription did not match the requested text closely enough "
            f"(word error rate {error_rate:.0%}; limit 25%)"
        )
    print("PASS: generated audio speaks the expected text")


if __name__ == "__main__":
    main()
