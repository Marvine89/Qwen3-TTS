# Transformers 5.19.0 smoke test

This small project runs the [Qwen3-TTS custom voice example](https://github.com/QwenLM/Qwen3-TTS)
against the checkout that contains the Transformers 5.19.0 compatibility changes. It loads
the 0.6B CustomVoice checkpoint with `Qwen3TTSModel.from_pretrained`, generates
a short English sample, checks the returned audio, transcribes it with Whisper Tiny,
and fails if the recognized words do not closely match the requested text.

The first run downloads the Qwen model and Whisper Tiny from Hugging Face. Both
models and the device can be changed with command-line options.

## Run

From this directory, create a Python 3.14 environment and install this checkout:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python smoke_test.py
```

By default, the script selects CUDA when available and otherwise uses CPU. To
select a device or a different Qwen3-TTS model explicitly:

```sh
python smoke_test.py --device cpu
python smoke_test.py --device cuda:0 --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The output defaults to `smoke_test_output.wav` in the current directory. A
successful run exits with status 0 and prints the Transformers version, model,
sample rate, audio duration, transcript, and word error rate. It exits with an
error if Transformers is not exactly 5.19.0, generation returns invalid audio,
or the transcript differs from the requested text by more than 25% word error.
Use `--seed` to make generation repeatable, `--asr-model` to select a different
speech recognition model, and `--expected-text` when the spoken wording differs
from the synthesis prompt.
