from mel import librosa_wav2spec
from model import HifiGAN
from utils.hparams import set_hparams
import numpy as np
from scipy.io import wavfile


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))



# file = 'gen.wav'
# set_hparams()
# wav_spec = librosa_wav2spec(file)
# vocoder = HifiGAN()
# mel = wav_spec['mel']

# wav_gen = vocoder.spec2wav(mel)
# save_wav(wav_gen, file.replace('.wav', '_gen.wav'), 16000)

set_hparams()
vocoder = HifiGAN()
#mel = np.load('D:/Research/seeg_speech/eeg2mel/output/result/LJSpeech/sub6_1_combined-00.npy')
#wav_gen = vocoder.spec2wav(mel)
#save_wav(wav_gen, 'D:/Research/seeg_speech/eeg2mel/output/result/LJSpeech/sub6_1_combined-00.wav', 16000)

import glob
count= 0
mel_list = glob.glob('/path/to/mel')
for mel_name in mel_list:
    print
    mel = np.load(mel_name)
    wav_gen = vocoder.spec2wav(mel)
    save_wav(wav_gen, mel_name.replace('.npy', '.wav'), 16000)
    print(mel_name)
    count += 1
    print(count)
