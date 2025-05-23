import os
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Function to load and resample a wav file
def load_and_resample(file_path, target_sr):
    y, sr = librosa.load(file_path, sr=None)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y

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


# Paths to the directories with the wav files
gt_dir = '/path/to/gt'
pred_dir = '/path/to/pred'

# Get the list of file names, assuming the filenames are consistent
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".wav")])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".wav")])

# Initialize list to store MCD values
mcd_values = []

# Compute MCD for each pair of files
for gt_name, pred_name in zip(gt_files, pred_files):
    print(gt_name)
    print(pred_name)
    # Load and resample the ground truth file to 16000 Hz
    gt_path = os.path.join(gt_dir, gt_name)
    y_gt = load_and_resample(gt_path, 16000)

    # Load the predicted file
    pred_path = os.path.join(pred_dir, pred_name)
    y_pred = load_and_resample(pred_path, 16000)

    # Cut the predicted file if it's longer than the ground truth
    if len(y_pred) > len(y_gt):
        y_pred = y_pred[:len(y_gt)]

    # Compute MFCCs
    mfcc_gt = librosa.feature.mfcc(y=y_gt, sr=16000, n_mfcc=25)
    mfcc_pred = librosa.feature.mfcc(y=y_pred, sr=16000, n_mfcc=25)

    mfcc_gt = mfcc_gt.T
    mfcc_pred = mfcc_pred.T
    
    print(mfcc_gt.shape)
    print(mfcc_pred.shape)
    
    # Calculate MCD
    mcd = mcd_calc(mfcc_gt, mfcc_pred)
    mcd_values.append(mcd)
    
    print(mcd)
    print()

# Output the average MCD
average_mcd = np.mean(mcd_values)
print('Average MCD:', average_mcd)


