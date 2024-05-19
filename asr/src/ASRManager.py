import whisper
import numpy as np
import text_hammer as th


class ASRManager:

    def __init__(self):
        # initialize the model here
        self.model = whisper.load_model('large')

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
            if words[i].isdigit():
                words[i] = ' '.join([num_to_word[digit] for digit in words[i]])

            # Split commonly found military terms
            if words[i] == "antiair":
                words[i] = "anti-air"
            elif words[i] == "surfacetoair":
                words[i] = "surface-to-air"

        # Join the words back into a sentence
        processed_pred = ' '.join(words).lower()

        return processed_pred

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        audio_signal = np.frombuffer(audio_bytes, dtype='<i2')
        # PREPROCESS AUDIO BYTES
        audio_signal = self.pre_process(audio_signal)
        # INFER
        result = self.model.transcribe(np.array(audio_signal, dtype=np.float32))
        # POSTPROCESS
        predicted_transcription = self.post_process(th.remove_special_chars(result["text"]))
        return predicted_transcription

if __name__ == "__main__":
    import wave
    a = ASRManager()
    with wave.open('D:/Github/til-24-base/advance/audio/audio_0.wav', 'rb') as wav_file:
        channels_number, sample_width, framerate, frames_number, compression_type, compression_name = wav_file.getparams()
        frames = wav_file.readframes(frames_number)
        print(a.transcribe(frames))
        print()
