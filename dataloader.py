import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class EEGDataset(Dataset):
    def __init__(self, eeg_files, label_files):
        self.eeg_files = eeg_files
        self.label_files = label_files
    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        label = np.load(self.label_files[idx])
        eeg_file = self.eeg_files[idx]

            
        eeg_data = torch.tensor(np.load(self.eeg_files[idx]), dtype=torch.float32)
        label_data = torch.tensor(label, dtype=torch.float32)
        ground_truth_text = os.path.basename(eeg_file).split('_')[0]
        return eeg_data, label_data,ground_truth_text

class EEGDataset_classify(Dataset):
    def __init__(self, eeg_files, label_files, class_mapping):
        self.eeg_files = eeg_files
        self.label_files = label_files
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        
        eeg_data = torch.tensor(np.load(self.eeg_files[idx]), dtype=torch.float32)
        label_file = self.label_files[idx]
        ground_truth_text = os.path.basename(label_file).split('_')[0]
        class_label = self.class_mapping[ground_truth_text]
        label_data = torch.tensor(class_label, dtype=torch.long)  
        
        return eeg_data, label_data,ground_truth_text
    
class EEGDataset_multi(Dataset):
    def __init__(self, eeg_files, label_files, class_mapping):
        self.eeg_files = eeg_files
        self.label_files = label_files
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        
        #eeg_T_prime = list(range(0,14)) + list(range(148,162))
        #print(eeg_T_prime)
        #eeg_data = torch.tensor(np.load(self.eeg_files[idx])[:,eeg_T_prime], dtype=torch.float32)
        eeg_data = torch.tensor(np.load(self.eeg_files[idx]), dtype=torch.float32)
        #print(eeg_data.shape)
        label_file = self.label_files[idx]
        label_mel = torch.tensor(np.load(label_file), dtype=torch.float32)
        ground_truth_text = os.path.basename(label_file).split('_')[0]
        class_label = self.class_mapping[ground_truth_text]
        label_class = torch.tensor(class_label, dtype=torch.long)  
        
        return eeg_data, label_mel, label_class, ground_truth_text