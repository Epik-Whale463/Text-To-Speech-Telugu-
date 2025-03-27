# Telugu Text-to-Speech

## Huggingface Model card
```https://huggingface.co/Epikwhale/speecht5_finetuned_telugu_charan```
![image](https://github.com/user-attachments/assets/7c54d8e8-dd6a-4f9d-aaf4-bebdd2b58c75)

This project implements a Text-to-Speech (TTS) system for the Telugu language by fine-tuning the Microsoft SpeechT5 model on Telugu audio data.

## Overview

This project:
- Uses the Microsoft SpeechT5 TTS model as a base
- Fine-tunes it on the IndicTTS_Telugu dataset from SPRINGLab
- Implements a transliteration approach to handle Telugu script characters
- Supports speaker embedding for voice style transfer

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Datasets
- SoundFile
- Accelerate
- SpeechBrain==0.5.16
- HyperPyYAML

## Dataset

The project uses the `SPRINGLab/IndicTTS_Telugu` dataset, which contains:
- 8,576 Telugu audio clips with corresponding text transcriptions
- Gender metadata for each speaker
- Audio sampled at 16kHz

## Model Architecture

The system utilizes:
- Microsoft's SpeechT5 architecture for text-to-speech generation
- SpeechBrain's speaker recognition (x-vector) model for speaker embeddings
- HiFiGAN vocoder for waveform generation

## Usage

To generate Telugu speech:

```python
# Load models
processor = SpeechT5Processor.from_pretrained("your_finetuned_model")
model = SpeechT5ForTextToSpeech.from_pretrained("your_finetuned_model")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Input Telugu text
text = "తెలుగు భాష అందమైన శబ్దంతో గొప్ప సంప్రదాయంతో విస్తారమైన సాహిత్యంతో లోకానికి వెలుగు చిందిస్తుంది"

# Process text and generate speech
inputs = processor(text=text, return_tensors="pt")
speaker_embeddings = ... # Load or create speaker embeddings
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# Save output
import soundfile as sf
sf.write('output.wav', speech.numpy(), 16000)
```

## Telugu Script Handling

The project implements a transliteration-based approach to handle Telugu script by mapping Telugu characters to Latin equivalents that the SpeechT5 tokenizer can process.

## Fine-tuning Process

The model training process includes:
1. Data preprocessing and tokenization
2. Speaker embedding extraction with SpeechBrain
3. Fine-tuning with gradient accumulation and checkpointing
4. Evaluation on a held-out test set

## Limitations

- Handling of numbers in Telugu requires conversion to word form
- Some complex phonetic patterns may not be perfectly captured
- Voice quality depends on the speaker embeddings used

## Future Work

- Expand the dataset with more diverse speakers
- Improve pronunciation of complex phonetic patterns
- Add support for more Indic languages
- Integrate with a text normalization system for handling numbers and dates

## Acknowledgements

- Microsoft for the SpeechT5 model
- SPRINGLab for the IndicTTS_Telugu dataset
- SpeechBrain for the speaker recognition models 
