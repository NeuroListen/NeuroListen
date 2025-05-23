import librosa
import numpy as np
import os
import re
import hashlib
import logging
import datetime
import torch
import sys
from scipy.spatial.distance import euclidean
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(base_dir)

sys.path.append(os.path.join(base_dir, 'src','wav2mel_intergrative'))


from mel_1 import librosa_wav2spec
from model_1 import HifiGAN
from utils_1.hparams import set_hparams
import numpy as np
from scipy.io import wavfile
from fastdtw import fastdtw
import pysptk
import pyworld
    

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
from pystoi.stoi import stoi
from scipy.io import wavfile
from scipy.signal import stft

class PCCLoss(nn.Module):
    def __init__(self):
        super(PCCLoss, self).__init__()

    def forward(self, outputs, labels):
        batch_size = outputs.shape[0]
        num_bins = outputs.shape[2]


        outputs = outputs.reshape(-1, num_bins)
        labels = labels.reshape(-1, num_bins)

        mean_output = torch.mean(outputs, dim=0)
        mean_label = torch.mean(labels, dim=0)
        std_output = torch.std(outputs, dim=0)
        std_label = torch.std(labels, dim=0)


        norm_output = (outputs - mean_output) / std_output
        norm_label = (labels - mean_label) / std_label


        correlation_matrix = torch.mean(norm_output * norm_label, dim=0)


        avg_correlation = torch.mean(correlation_matrix)

        pcc_loss = 1 - avg_correlation
        return pcc_loss


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav = wav * 32767
    wavfile.write(path[:-4] + '.wav', sr, wav.astype(np.int16))


def split_dataset(eeg_dir, label_dir, seed=42, repeat_threshold=10, k=5, mode = 'test_less'):
    """
    Splits the dataset with the following considerations:
    - All samples of words exceeding the repeat_threshold go to the training set.
    - Attempts to select up to 'k' unique words for the test set, ensuring only one 
      sample per word in the test set.
    - Any other samples from the selected test words not chosen for the test set, as well
      as words not selected, go into the training set.

    Parameters:
    - eeg_dir: Directory with EEG files.
    - label_dir: Directory with label files.
    - seed: Seed for random operations, for reproducibility.
    - repeat_threshold: Threshold for deciding all samples of a word go to training set.
    - k: Maximum number of unique words to try including in the test set, one sample per word.

    Returns:
    - train_eeg_files, test_eeg_files, train_label_files, test_label_files
    """
    random.seed(seed)  # Seed for reproducibility

    # Gather and sort files
    eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])
    #print(len(eeg_files))
    #print(len(label_files))
    assert len(eeg_files) == len(label_files), "Mismatch in EEG and label files counts."

    # Extract word base names for grouping
    words = [os.path.splitext(os.path.basename(f))[0].split('_')[4] for f in label_files]
    #print(words)
    
    files_by_word = {}
    for word, eeg_file, label_file in zip(words, eeg_files, label_files):
        files_by_word.setdefault(word, []).append((eeg_file, label_file))
    
    # Initialize lists for dataset split
    train_eeg_files, test_eeg_files, train_label_files, test_label_files = [], [], [], []

    # Determine eligible words for the test set
    if mode == "test_less":
        eligible_words = [word for word, files in files_by_word.items() if len(files) <= repeat_threshold]
        eligible_words = [word for word, files in files_by_word.items()]
        print(eligible_words)
    elif mode == "test_more":
        eligible_words = [word for word, files in files_by_word.items() if len(files) >= repeat_threshold]

    # Randomly select up to 'k' words for the test set, ensuring diversity
    test_words = random.sample(eligible_words, min(k, len(eligible_words)))

    for word, files in files_by_word.items():
        if word in test_words:
            # Randomly choose one sample for the test set
            test_sample = random.choice(files)
            test_eeg_files.append(test_sample[0])
            test_label_files.append(test_sample[1])
            # Add remaining samples of the word to the training set
            for eeg_file, label_file in files:
                if (eeg_file, label_file) != test_sample:
                    train_eeg_files.append(eeg_file)
                    train_label_files.append(label_file)
        else:
            # Add all samples of the word to the training set
            for eeg_file, label_file in files:
                train_eeg_files.append(eeg_file)
                train_label_files.append(label_file)

    return train_eeg_files, test_eeg_files, train_label_files, test_label_files


def split_word_dataset_folds(eeg_dir, label_dir, seed=42):
    
    fold_list = []
    
    # Gather and sort files
    eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])
    assert len(eeg_files) == len(label_files), "Mismatch in EEG and label files counts."

    # Extract word base names for grouping
    words = [os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in label_files]
    files_by_word = {}
    for word, eeg_file, label_file in zip(words, eeg_files, label_files):
        files_by_word.setdefault(word, []).append((eeg_file, label_file))
        
    np.random.seed(42)
    word_test_indices = {word: np.random.permutation(len(files)).tolist() for word, files in files_by_word.items()}

    for fold_num in range(6):
    
        fold_files = []
        # Initialize lists for dataset split
        train_eeg_files, test_eeg_files, train_label_files, test_label_files = [], [], [], []

        for word, files in files_by_word.items():
            test_index = word_test_indices[word][fold_num % len(files)]
            test_sample = files[test_index]
            test_eeg_files.append(test_sample[0])
            test_label_files.append(test_sample[1])
            # Add remaining samples of the word to the training set
            for eeg_file, label_file in files:
                if (eeg_file, label_file) != test_sample:
                    train_eeg_files.append(eeg_file)
                    train_label_files.append(label_file)


        fold_files.append(train_eeg_files)
        fold_files.append(test_eeg_files)
        fold_files.append(train_label_files)
        fold_files.append(test_label_files)
        fold_list.append(fold_files)

    return fold_list


def split_sentence_dataset_folds(eeg_dir, label_dir, seed=42):
    
    fold_list = []
    
    # Gather and sort files
    eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])
    assert len(eeg_files) == len(label_files), "Mismatch in EEG and label files counts."

    # Extract word base names for grouping
    words = [os.path.splitext(os.path.basename(f))[0].split('_')[0] for f in label_files]
    files_by_word = {}
    for word, eeg_file, label_file in zip(words, eeg_files, label_files):
        files_by_word.setdefault(word, []).append((eeg_file, label_file))

    for fold_num in range(6):
        
        if fold_num % 3 == 0:
            start = 0
            end = 33
        elif fold_num % 3 == 1:
            start = 33
            end = 66
        else:
            start = 66
            end = 100
        
        if fold_num < 3:
            take = 0
        else:
            take = 1
            
        test_words = list(files_by_word.keys())[start:end]
        
        fold_files = []
        # Initialize lists for dataset split
        train_eeg_files, test_eeg_files, train_label_files, test_label_files = [], [], [], []

        for word, files in files_by_word.items():
            if word in test_words:
                test_sample = files[take]
                test_eeg_files.append(test_sample[0])
                test_label_files.append(test_sample[1])
                # Add remaining samples of the word to the training set
                for eeg_file, label_file in files:
                    if (eeg_file, label_file) != test_sample:
                        train_eeg_files.append(eeg_file)
                        train_label_files.append(label_file)
            else:
                # Add all samples of the word to the training set
                for eeg_file, label_file in files:
                    train_eeg_files.append(eeg_file)
                    train_label_files.append(label_file)

        fold_files.append(train_eeg_files)
        fold_files.append(test_eeg_files)
        fold_files.append(train_label_files)
        fold_files.append(test_label_files)
        fold_list.append(fold_files)

    return fold_list


def mel_to_wav(mel, num_bins):
    if num_bins == 80:
        set_hparams(config='/root/autodl-tmp/src/wav2mel_intergrative/config/hifigan.yaml')
    if num_bins == 13:
        set_hparams(config='../wav2mel_intergrative/config/hifigan_23.yaml')
    vocoder = HifiGAN()

    wav_gen = vocoder.spec2wav(mel)

    return wav_gen
    

        
        
        

def crop_and_save_data(eeg_data_dir='/path/to/eeg', 
                       label_data_dir='/path/to/label', 
                       eeg_save_dir='eeg2unit/data/eeg', 
                       label_save_dir='eeg2unit/data/label', 
                       eeg_shape=(400,252), 
                       label_shape=(100,)):
    
    eeg_files = [os.path.join(eeg_data_dir, f) for f in os.listdir(eeg_data_dir) if f.endswith('.npy')]
    label_files = [os.path.join(label_data_dir, f) for f in os.listdir(label_data_dir) if f.endswith('.npy')]
    
    eeg_files.sort()
    print(eeg_files)
    label_files.sort()
    print(label_files)
    
    if not os.path.exists(eeg_save_dir):
        os.makedirs(eeg_save_dir)
    
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    
    total_files = len(eeg_files)
    
    for file_index, (eeg_file, label_file) in enumerate(zip(eeg_files, label_files)):
        eeg_data = np.load(eeg_file)
        label_data = np.load(label_file)
        
        # Print the shapes of the current files
        print(f"Processing file {file_index + 1}/{total_files}")
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Label data shape: {label_data.shape}")
        
        eeg_crops_possible = eeg_data.shape[0] // eeg_shape[0]
        label_crops_possible = label_data.shape[0] // label_shape[0]
        
        num_crops = min(eeg_crops_possible, label_crops_possible)
        
        for i in range(num_crops):
            start_idx_eeg = i * eeg_shape[0]
            start_idx_label = i * label_shape[0]
            
            # Print the cropping points
            print(f"Cropping EEG data from index {start_idx_eeg} to {start_idx_eeg + eeg_shape[0]}")
            print(f"Cropping label data from index {start_idx_label} to {start_idx_label + label_shape[0]}")
            
            cropped_eeg = eeg_data[start_idx_eeg:start_idx_eeg+eeg_shape[0]]
            cropped_label = label_data[start_idx_label:start_idx_label+label_shape[0]]
            
            print(f"EEG data shape after cropping: {cropped_eeg.shape}")
            print(f"Label data shape after cropping: {cropped_label.shape}")
            base_name = os.path.basename(eeg_file).replace('.npy', '')
            np.save(os.path.join(eeg_save_dir, f"{base_name}_crop_{i}.npy"), cropped_eeg)
            
            base_name = os.path.basename(label_file).replace('.npy', '')
            np.save(os.path.join(label_save_dir, f"{base_name}_crop_{i}.npy"), cropped_label)
        
        # Print progress (进度)
        progress_percentage = (file_index + 1) / total_files * 100
        print(f"进度: {progress_percentage:.2f}%")



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_hyperparameters_from_log(log_path):
    hyperparameters = {}
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
            # Find the start and end indices for hyperparameters in the log
            start_index = None
            end_index = None
            for i, line in enumerate(lines):
                if "Training with the following hyperparameters:" in line:
                    start_index = i
                if "preparing for training..." in line:
                    end_index = i
                    break
            
            # If start and end indices are found, parse the hyperparameters
            if start_index is not None and end_index is not None:
                for line in lines[start_index+1:end_index]:
                    if "INFO" in line:
                        parts = line.split('- INFO -')
                        if len(parts) > 1:
                            info = parts[1].strip()
                            if ':' in info:
                                key, value = info.split(':', 1)
                                key = key.strip()
                                value = eval(value.strip())  # Convert string representation to actual value
                                hyperparameters[key] = value
    except Exception as e:
        print(f"Error retrieving hyperparameters from log: {e}")
    
    return hyperparameters


import re




def custom_sort_key(filename):
    # Extract the numbers from the filename
    sub_match = re.search(r'sub(\d+)', filename)
    sub_num = int(sub_match.group(1)) if sub_match else 0

    sub_sub_match = re.search(r'sub\d+_(\d+)', filename)
    sub_sub_num = int(sub_sub_match.group(1)) if sub_sub_match else 0

    crop_match = re.search(r'crop_(\d+)', filename)
    crop_num = int(crop_match.group(1)) if crop_match else 0

    return (sub_num, sub_sub_num, crop_num)

def sort_files(files):
    return sorted(files, key=custom_sort_key)




def hyperparameters_to_string(hyperparameters):
    return '_'.join(f"{k}={v}" for k, v in sorted(hyperparameters.items()))



# Function to get MD5 hash of a string
def get_md5_hash(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()



def setup_logging(base_dir, hyperparameters_string,current_time):
    log_dir = os.path.join(base_dir, 'train_logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    hash_string = get_md5_hash(hyperparameters_string)
    hash_string= current_time+"_"+hash_string
    log_subdir = os.path.join(log_dir, hash_string)
    if not os.path.exists(log_subdir):
        os.makedirs(log_subdir)

    log_filename = os.path.join(log_subdir, f"training_log_{current_time}.txt")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_filename

def setup_logging_handler(base_dir, hyperparameters_string,current_time):
    log_dir = os.path.join(base_dir, 'train_logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    hash_string = get_md5_hash(hyperparameters_string)
    hash_string= current_time+"_"+hash_string
    log_subdir = os.path.join(log_dir, hash_string)
    if not os.path.exists(log_subdir):
        os.makedirs(log_subdir)

    log_filename = os.path.join(log_subdir, f"training_log_{current_time}.txt")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


    return log_filename, file_handler

def close_logging(file_handler):
    root_logger = logging.getLogger()
    root_logger.removeHandler(file_handler)
    file_handler.close()
    

def setup_model_saving(base_dir, hyperparameters_string, current_time, global_step= None):
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    #print(model_dir)

    hash_string = get_md5_hash(hyperparameters_string)
    hash_string= str(current_time)+"_"+hash_string
    model_subdir = os.path.join(model_dir, hash_string)
    if not os.path.exists(model_subdir):
        os.makedirs(model_subdir)
        
    #print(model_subdir)
    if global_step is not None:
        model_filename = os.path.join(model_subdir, f"model_{current_time}_{global_step}.pth")
    else:
        model_filename = os.path.join(model_subdir, f"best_model_{current_time}.pth")

    return model_filename


def get_weight(dataloader, device):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.view(-1).cpu().numpy())  # Flatten the labels and convert to numpy
    class_weights = compute_class_weights(all_labels)
    return torch.FloatTensor(class_weights).to(device)


def compute_class_weights(labels, epsilon=1e-10):
    # Ensure labels is a 1D numpy array
    labels = np.array(labels).flatten()
    
    # Compute class counts
    class_counts = np.bincount(labels)
    
    # Find the class with the maximum occurrences
    most_common_class = np.argmax(class_counts)
    most_common_class_count = class_counts[most_common_class]
    
    logging.info(f"The class with the most occurrences is {most_common_class} with {most_common_class_count} occurrences.")
    
    # Log classes based on their frequency
    sorted_indices = np.argsort(class_counts)[::-1]  # Sort in descending order
    for rank, idx in enumerate(sorted_indices, 1):
        logging.info(f"Rank {rank}: Class {idx} with {class_counts[idx]} occurrences (Frequency: {class_counts[idx]/len(labels)*100:.2f}%).")
    
    # Compute class weights
    class_weights = most_common_class_count / (class_counts + epsilon)
    return class_weights

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


def create_class_mapping(files):
    class_mapping = {}
    class_index = 0
    for file in files:
        #sub 6 is 5, sub5 is 4, sub2 is 2
        label = os.path.basename(file).split('_')[0]
        if label not in class_mapping:
            class_mapping[label] = class_index
            class_index += 1
    return class_mapping
    

def create_second_mapping(files):
    class_mapping = {}
    for file in files:
        label = os.path.basename(file).split('_')[4]
        if '\u4e00' <= label <= '\u9fff':
            class_mapping[label] = 1
        else:
            class_mapping[label] = 0
    return class_mapping

def correlation_persample(outputs, labels):
    
    # input: a list with num_batch (bs, time, bin)s
    
    all_outputs = np.concatenate(outputs, axis=0) # (num_samples, time, bin)
    all_labels = np.concatenate(labels, axis=0)
    correlations_per_sample = []
    
    for output, label in zip(all_outputs, all_labels):

        for bin in range(output.shape[1]):
            correlation,_ = pearsonr(output[:, bin], label[:, bin])
            correlations_per_sample.append(correlation)        


    return np.mean(correlations_per_sample)


def correlation_conca(outputs, labels):
    
    # input: a list with num_batch (bs, time, bin)s
    
    all_outputs = np.concatenate(outputs, axis=0) # (num_samples, time, bin)
    all_labels = np.concatenate(labels, axis=0)
    
    all_outputs = all_outputs.reshape(-1, all_outputs.shape[2])
    all_labels = all_labels.reshape(-1, all_labels.shape[2])
    
    correlation_per_bin = []
    
    for bin in range(all_outputs.shape[1]):
        correlation, _ = pearsonr(all_outputs[:, bin], all_labels[:, bin])
        correlation_per_bin.append(correlation)
        
    return np.mean(correlation_per_bin)


def correlation_persample_nobatch(outputs, labels):
    
    # input:  (time, bin)
    
    correlation_per_bin = []

    for bin in range(outputs.shape[1]):
        correlation, _ = pearsonr(outputs[:,bin], labels[:,bin])
        #logging.info(f"correlation for {bin}:{correlation}")
        
        correlation_per_bin.append(correlation)
    
    return np.mean(correlation_per_bin)


def get_warping_path(query_path, reference_path):

    interp_func = interp1d(query_path, reference_path, kind="linear")
    warping_index = interp_func(np.arange(query_path.min(), reference_path.max() + 1)).astype(np.int64)
    warping_index[0] = reference_path.min()

    return warping_index


def dtw_warping(query_spec, reference):
    distance, path = fastdtw(query_spec, reference, dist=euclidean, radius=len(query_spec))
    query, ref = zip(*path)
    query, ref = np.array(query), np.array(ref)
    warping_indices = get_warping_path(query, ref)
    return reference[warping_indices]


def correlation_persample_dtw(outputs, labels):
    
    # input: a list with num_batch (bs, time, bin)s
    
    all_outputs = np.concatenate(outputs, axis=0) # (num_samples, time, bin)
    all_labels = np.concatenate(labels, axis=0)
    correlations_per_sample = []
    
    for output, label in zip(all_outputs, all_labels):
        
        warped_ref = dtw_warping(output, label)

        for bin in range(output.shape[1]):
            correlation,_ = pearsonr(output[:, bin], warped_ref[:, bin])
            correlations_per_sample.append(correlation)        


    return np.mean(correlations_per_sample)


def correlation_conca_dtw(outputs, labels):
    
    # input: a list with num_batch (bs, time, bin)s
    
    all_outputs = np.concatenate(outputs, axis=0) # (num_samples, time, bin)
    all_labels = np.concatenate(labels, axis=0)
    
    all_outputs = all_outputs.reshape(-1, all_outputs.shape[2])
    all_labels = all_labels.reshape(-1, all_labels.shape[2])
    
    correlation_per_bin = []
    
    warped_ref = dtw_warping(all_outputs, all_labels)
    
    for bin in range(all_outputs.shape[1]):
        correlation, _ = pearsonr(all_outputs[:, bin], warped_ref[:, bin])
        correlation_per_bin.append(correlation)
        
    return np.mean(correlation_per_bin)

def compute_mse_metric(outputs, labels):
    all_outputs = np.concatenate(outputs, axis=0)  # [num_samples, time, bin]
    all_labels = np.concatenate(labels, axis=0)

    rmse_list = []
    for output, label in zip(all_outputs, all_labels):
        mse = mean_squared_error(label.flatten(), output.flatten())
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

    return np.mean(rmse_list)



def compute_stoi_metric(x, y, fs):
    win_len = int(fs * 0.025)  # 窗长为25ms
    hop_len = int(fs * 0.010)  # 窗移为10ms

    _, _, Pxo = stft(x, fs=fs, nperseg=win_len, noverlap=hop_len)
    _, _, Pyo = stft(y, fs=fs, nperseg=win_len, noverlap=hop_len)

    stoi_values = []
    for i in range(Pxo.shape[1]):
        Pxo_i = np.abs(Pxo[:, i])
        Pyo_i = np.abs(Pyo[:, i])

        Rxy = np.sum(Pxo_i * Pyo_i) / np.sqrt(np.sum(Pxo_i ** 2) * np.sum(Pyo_i ** 2))
        stoi_values.append(Rxy)

    return np.mean(stoi_values)
