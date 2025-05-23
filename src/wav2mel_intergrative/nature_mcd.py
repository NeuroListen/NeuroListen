import librosa
import numpy as np
import pysptk
import pyworld
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os


def normalize_volume(audio):
    """ Normalize an audio waveform to be between 0 and 1 """
    rms = librosa.feature.rms(y=audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def mcd_calc(C, C_hat):
    """ Computes MCD between ground truth and target MFCCs. First computes DTW aligned MFCCs

    Consistent with Anumanchipalli et al. 2019 Nature, we use MC 0 < d < 25 with k = 10 / log10
    """

    # ignore first MFCC
    K = 10 / np.log(10)
    C = C[:, 1:25]
    C_hat = C_hat[:, 1:25]

    # compute alignment
    distance, path = fastdtw(C, C_hat, dist=euclidean)
    distance/= (len(C) + len(C_hat))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    C, C_hat = C[pathx], C_hat[pathy]
    frames = C_hat.shape[0]

    # compute MCD
    z = C_hat - C
    s = np.sqrt((z * z).sum(-1)).sum()
    MCD_value = K * float(s) / float(frames)
    return MCD_value

def wav2mcep_numpy(wav, sr, alpha=0.65, fft_size=512, mcep_size=25):
    """ Given a waveform, extract the MCEP features """

    # Use WORLD vocoder to extract spectral envelope
    _, sp, _ = pyworld.wav2world(wav.astype(np.double), fs=sr,frame_period=5.0, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mgc

def compute_mcd(sample, y, sr_desired=16000):
    """ Computes MCD between target waveform and predicted waveform """

    # equalize lengths
    if len(sample) < len(y):
        y = y[:len(sample)]
    else:
        sample = sample[:len(y)]

    # normalize volume
    y = normalize_volume(y)
    sample = normalize_volume(sample)

    # compute MCD
    mfcc_y_ = wav2mcep_numpy(sample, sr_desired)
    mfcc_y = wav2mcep_numpy(y, sr_desired)

    mcd = mcd_calc(mfcc_y, mfcc_y_)
    return mcd


ground_truth_path = '/path/to/gt'
predicted_path = '/path/to/pred'


gt_files = sorted([f for f in os.listdir(ground_truth_path) if f.endswith('.wav')])
pred_files = [f.replace(".wav", "_mel.wav") for f in gt_files]

mcd_values = []
for gt_file, pred_file in zip(gt_files, pred_files):
    gt_file_path = os.path.join(ground_truth_path, gt_file)
    pred_file_path = os.path.join(predicted_path, pred_file)
    

    gt_audio, gt_sr = librosa.load(gt_file_path, sr=16000)
    pred_audio, pred_sr = librosa.load(pred_file_path, sr=16000)


    mcd_value = compute_mcd(gt_audio, pred_audio, gt_sr)
    print(f"MCD for {gt_file} and {pred_file}: {mcd_value}")
    mcd_values.append([mcd_value,pred_file])

print("Average MCD: ", np.mean([x[0] for x in mcd_values]))

mcd_values.sort(key=lambda x: x[0])
for i in mcd_values:
    if i[0] < 5.5:
        print(i)

good_mcd = [x[1] for x in mcd_values if x[0] < 5.5 ]

for f in good_mcd:
    pred_file_path = os.path.join(predicted_path, f)
    dest_path = os.path.join(os.path.dirname(os.path.dirname(predicted_path)),"good_mcd/")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)  
    os.system(f"cp {pred_file_path} {dest_path}")

