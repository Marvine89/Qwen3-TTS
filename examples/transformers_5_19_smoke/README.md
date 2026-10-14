# Transformers 5.19.0 smoke test

This small project runs the [Qwen3-TTS custom voice example](https://github.com/QwenLM/Qwen3-TTS)
against the checkout that contains the Transformers 5.19.0 compatibility changes. It loads
the 0.6B CustomVoice checkpoint with `Qwen3TTSModel.from_pretrained`, generates
a short English sample, checks the returned audio, and writes a WAV file.

The first run downloads the model weights from Hugging Face. The model and
device can be changed with command-line options.

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
sample rate, audio duration, and output path. It exits with an error if
Transformers is not exactly 5.19.0 or if generation returns invalid audio.
