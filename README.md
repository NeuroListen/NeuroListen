# Listening to the Brain: Multi-Band sEEG Auditory Reconstruction via Dynamic Spatio-Temporal Hypergraphs
**NeuroListen** is the first publicly available stereotactic EEG (sEEG) dataset designed for auditory reconstruction. The dataset includes neural recordings collected while participants listen Chinese Mandarin words, English words, and Chinese Mandarin digits with corresponding auditory signals.

This repository provides the dataset, and code for running experiments on auditory reconstruction using the **HyperSpeech** framework.
---
## **Environment Setup**

To run the experiments and use the dataset, follow these steps to set up the environment.

### **Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/NeuroListen/NeuroListen
cd NeuroListen
```

### Install Dependencies
Make sure you have Python 3.7+ installed. It’s recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

## sEEG Preprocess

```bash
python preprocess_eeg.py
```

## Model Training

```bash
python train_cnnlstm.py
```


## **Dataset**

The **NeuroListen** dataset is available for download. The dataset includes sEEG recordings and their corresponding audio samples for five subjects (Mandarin and English). Each recording contains speech data across various tasks (Mandarin words, English words, and Chinese Mandarin digits).

You can download the dataset from the following links:
- [Dataset Link](https://drive.google.com/drive/folders/1bw5OxA_cIPaR-aDgjsmmTd2FaZyhmbv0?usp=drive_link)
- [GitHub Dataset](https://github.com/NeuroListen/NeuroListen)

### **Dataset Structure**
The dataset is organized as follows:
```bash

NeuroListen/
    ├── BBS/
    │   ├── SUB1/
    │   ├── SUB2/
    │   ├── SUB3/
    │   ├── SUB4/
    │   └── SUB5/
    ├── HGA/
    │   ├── SUB1/
    │   ├── SUB2/
    │   ├── SUB3/
    │   ├── SUB4/
    │   └── SUB5/
    ├── LFS/
    │   ├── SUB1/
    │   ├── SUB2/
    │   ├── SUB3/
    │   ├── SUB4/
    │   └── SUB5/
    ├── MEL/
    │   ├── SUB1/
    │   ├── SUB2/
    │   ├── SUB3/
    │   ├── SUB4/
    │   └── SUB5/



```
- **MEL**: Mel-spectrograms of the audio.
- **BBS/HGA/LFS**: Different frequency bands of sEEG signals used for auditory reconstruction (Broadband, High Gamma Activity, Low-Frequency Signals).



