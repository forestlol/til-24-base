import whisperx
import numpy as np
import torch
import re
import os

class ASRManager:

    def __init__(self):
        # initialize the model here
        self.device = "cuda"
        self.batch_size = 128  # reduce if low on GPU mem
        self.compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
        # print("CUDA: " + str(torch.cuda.is_available()))
        print("Loading VAD Model")
        vad_model = whisperx.vad.load_vad_model(
            torch.device(self.device),
            model_fp=os.path.join("models", "whisperx-vad-segmentation.bin"),
        )

        print("Loading Whisper Model")
        self.model = whisperx.load_model(whisper_arch=os.path.join(".", "models", "distill-large-v3"), device=self.device, language="en", compute_type=self.compute_type, vad_model=vad_model)
        print("Done Loading")

    def remove_punctuation_regex(self, s):
        return re.sub(r'[^\w\s]', '', s)

    def pre_process(self, audio):
        return audio

    def post_process(self, pred):
        # Mapping of numbers to their word form
        num_to_word = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

        # Split the sentence into words
        words = pred.split()
        
        # Process each word
        for i in range(len(words)):
            # If the word is a number, convert it to word form
            if any(ch.isdigit() for ch in words[i]):
                arr = []
                for ch in words[i]:
                    if ch.isdigit():
                        arr.append(num_to_word[ch])
                    else:
                        arr.append(ch)
                words[i] = ' '.join(arr)

            # Split commonly found military terms
            if words[i] == "antiair":
                words[i] = "anti-air"
            elif words[i] == "surfacetoair":
                words[i] = "surface-to-air"
            elif words[i] == "emb":
                words[i] = "emp"

        # Join the words back into a sentence
        processed_pred = ' '.join(words).lower()

        return processed_pred

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio_signal = np.frombuffer(audio_bytes, dtype='<i2')
        # PREPROCESS AUDIO BYTES
        audio_signal = self.pre_process(audio_signal)
        # INFER
        audio_array = np.array(audio_signal, dtype=np.float32)

        segments = self.model.transcribe(audio_array, language="en", batch_size=self.batch_size)
        transcription = "".join(segment['text'] for segment in segments['segments']).strip()

        # POSTPROCESS
        predicted_transcription = self.post_process(self.remove_punctuation_regex(transcription))
        print(f"| Pred: {predicted_transcription}")
        return predicted_transcription


if __name__ == '__main__':
    import wave
    a = ASRManager()
    with wave.open('D:/Github/til-24-base/advance/audio/audio_0.wav', 'rb') as wav_file:
        channels_number, sample_width, framerate, frames_number, compression_type, compression_name = wav_file.getparams()
        frames = wav_file.readframes(frames_number)
        print(a.transcribe(frames))
        print()
