# -*- coding: gbk -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from dataloader import EEGDataset, EEGDataset_multi

from model import CustomLoss, EEGModelMultiAggr1
from torch.utils.data import DataLoader
import logging
import datetime
import utils
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from datetime import timedelta
from utils import split_dataset,PCCLoss, correlation_conca, correlation_persample,correlation_persample_nobatch,correlation_conca_dtw,correlation_persample_dtw,create_class_mapping


random_seed = 0


print('training on model')


# Hyperparameters
hyperparameters = {
    # optimizer
     "batch_size":16,
    "learning_rate": 1.0e-4,
    "num_epochs": 100, # 4000
    "milestones": [20000], # not used now
    "gamma": 0.5,
    "grad_norm": 1,

    # training
    "epochs_within_evaluation": 10, # 500
    "epochs_per_checkpoint":100, #1000
    "patience": -1,
    "test_mode": "test_less",
    "test_size":30,
    "repeat_threshold": 100,

    # model
    "n_classes": 80,
    "cnn_dim": 64, #  64 260
    "rnn_dim": 256,# 256 130
    "num_rnn_layers": 3,
    "dropout_cnn_pre": 0.7,
    "dropout_rnn": 0.7,
    "bidirectional": True,

    "in_channels": 202,
    "dropout_cnn_post": 0.7,
    "bidirectional": True,
    "KS": 4,
    "num_transformer_layers": 8,
    "num_transformer_heads": 10,

    "rnn_type": "biLSTM",
    "loss": "custom",
    "alpha":0.1,
    "relu1": True,
    "sigmoid1": False,

}

base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
base_dir = os.path.join(base, "exp")

hyperparameters_string = utils.hyperparameters_to_string(hyperparameters)
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Setup logging
log_filename = utils.setup_logging(base_dir, hyperparameters_string, current_time)

# Setup model saving
model_save_path = utils.setup_model_saving(base_dir, hyperparameters_string, current_time)
logging.info(f"Model will be saved to: {model_save_path}")

# Notify the user that data is being loaded
logging.info("Loading EEG data files...")


base ="dataset_nn"



eeg_dir = '/path/to/eeg_folder'
label_dir = '/path/to/mel_folder'

logging.info(f"eeg_dir: {eeg_dir}")
logging.info(f"label_dir: {label_dir}")


eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith('.npy')])
label_files = []
for eeg_file in eeg_files:
    word = eeg_file.split('_')[-3]
    word_type = eeg_file.split('_')[-2]
    for label_file in os.listdir(label_dir):
        if word in label_file and word_type in label_file:
            label_files.append(os.path.join(label_dir, label_file))
            break



#label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.npy')])


#print(len(eeg_files), len(label_files))
assert len(eeg_files) == len(label_files)
train_eeg_files, test_eeg_files, train_label_files, test_label_files = train_test_split(eeg_files, label_files, test_size=0.2, random_state=random_seed)
print(len(train_eeg_files), len(test_eeg_files), len(train_label_files), len(test_label_files))



label_to_class = create_class_mapping(train_label_files + test_label_files)
num_classes = len(label_to_class)

logging.info(f"Number of classes: {num_classes}")
logging.info(f"Label to class mapping: {label_to_class}")

#logging.info(f"eeg_dir: {eeg_dir},label_dir: {label_dir}")

# Extract just the file names for logging
train_eeg_file_names = [os.path.basename(path) for path in train_eeg_files]
train_label_file_names = [os.path.basename(path) for path in train_label_files]
test_eeg_file_names = [os.path.basename(path) for path in test_eeg_files]
test_label_file_names = [os.path.basename(path) for path in test_label_files]



# Notify the user that datasets are being created
logging.info("Creating EEG datasets...")

train_dataset = EEGDataset_multi(train_eeg_files, train_label_files, label_to_class)
test_dataset = EEGDataset_multi(test_eeg_files, test_label_files, label_to_class)

# Notify the user that data loaders are being created
logging.info("Creating data loaders...")

train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,num_workers=1)

for eeg_data, label_mel, label_class, ground_truth_texts in train_loader:
    logging.info("EEG data batch size: %s", eeg_data.size())
    logging.info("Label mel batch size: %s", label_mel.size())
    logging.info("Label class batch size: %s", label_class.size())
    logging.info("Ground truth texts: %s", ground_truth_texts)
    logging.info("Label class: %s", label_class)
    break  # we only want to look at the first batch

# Notify the user that data loading is complete
logging.info("EEG data loaded successfully!")

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' )

model = EEGModelMultiAggr1(
    in_channels=hyperparameters['in_channels'],
    cnn_dim=hyperparameters['cnn_dim'],
    rnn_dim=hyperparameters['rnn_dim'],
    KS=hyperparameters['KS'],
    num_rnn_layers=hyperparameters['num_rnn_layers'],
    num_transformer_layers=hyperparameters['num_transformer_layers'],
    num_transformer_heads=hyperparameters['num_transformer_heads'],
    dropout_cnn_pre=hyperparameters['dropout_cnn_pre'],
    dropout_rnn=hyperparameters['dropout_rnn'],
    bidirectional=hyperparameters['bidirectional'],
    n_classes=hyperparameters['n_classes'],
    rnn_type=hyperparameters['rnn_type'],
    relu1 = hyperparameters['relu1'],
    sigmoid1= hyperparameters['sigmoid1']
).to(device)


#model = EEGTransformerModel(in_channels=hyperparameters['in_channels']).to(device)

# Log the model structure
logging.info("Model Structure:")
logging.info(str(model))

logging.info("Model's layers and their corresponding number of parameters:")
total_params = 0
for name, parameter in model.named_parameters():
    if parameter.requires_grad:  # Checking if the parameter is trainable
        num_params = parameter.numel()  # Number of elements in the parameter
        total_params += num_params
        logging.info(f"{name}: {num_params}")

logging.info(f"Total number of trainable parameters: {total_params}")

# Count and log the number of trainable parameters
num_trainable_params = utils.count_trainable_parameters(model)
logging.info(f"Number of trainable parameters: {num_trainable_params}")

logging.info(f"__________________________________")
# Log the hyperparameters
logging.info("Training with the following hyperparameters:")
for key, value in hyperparameters.items():
    logging.info(f"{key}: {value}")


logging.info(f"__________________________________")
logging.info("preparing for training...")


# loss and run

if hyperparameters['loss'] == "MSE":
    loss_fn_reconstruction = nn.MSELoss()
elif hyperparameters['loss'] == "L1":
    loss_fn_reconstruction = nn.L1Loss()
elif hyperparameters['loss'] == "custom":
    loss_fn_reconstruction = CustomLoss(lambda_value=0.1)

logging.info(f"Loss function: {loss_fn_reconstruction}")

loss_fn_classification = nn.CrossEntropyLoss()

alpha = hyperparameters['alpha']


optimizer = optim.Adam(model.parameters(),
                       lr=hyperparameters['learning_rate'])
scheduler = MultiStepLR(optimizer, milestones=hyperparameters['milestones'], gamma=hyperparameters['gamma'])

best_loss = float('inf')
counter = 0

logging.info("Model starting training...")
training_start_time = time.time()

global_step = 0

for epoch in range(hyperparameters['num_epochs']):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    running_loss_reconstruction = 0.0
    running_loss_classification = 0.0
    val_check = False
    batch_start_time = time.time()


    for i, (eeg_data, labels, labels_class, _) in enumerate(train_loader):

        global_step += 1  # Increment at the start of each loop

        eeg_data, labels, labels_class = eeg_data.to(device), labels.to(device), labels_class.to(device)
        outputs, class_outputs= model(eeg_data)
        reconstruction_loss = loss_fn_reconstruction(outputs, labels)
        classification_loss = loss_fn_classification(class_outputs, labels_class)
        loss = reconstruction_loss + alpha * classification_loss
        #loss = classfication_loss

        running_loss += loss.item()
        running_loss_reconstruction += reconstruction_loss.item()
        running_loss_classification += classification_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['grad_norm'])
        optimizer.step()


        # Checkpoint and Evaluation Logic at Specified Intervals
        if ((epoch+1) % hyperparameters['epochs_per_checkpoint'] == 0) and (i == len(train_loader)-1):
            logging.info(f"__________________________________")
            logging.info("Started: Saving model checkpoint...")

            # Pre-check for output directory creation
            output_path = os.path.join(base_dir, 'output')
            os.makedirs(output_path, exist_ok=True)

            hash_string = utils.get_md5_hash(hyperparameters_string)
            hash_string= str(current_time)+"_"+hash_string

            out_subdir = os.path.join(output_path, hash_string)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)

            model.eval()
            val_loss = 0.0
            correlations = {'train': [], 'val': []}  # To store Pearson correlations
            mcds = {'train': [], 'val': []}

            # Unified handling for both datasets
            for data_subset, loader in [("train", train_loader), ("val", test_loader)]:
                subset_loss = 0.0
                subset_loss_class = 0.0
                subset_loss_rec = 0.0
                total = 0
                correct = 0
                top5_correct = 0

                with torch.no_grad():
                    all_outputs = []
                    all_labels = []

                    for j, (eeg_data_subset, labels_subset, labels_class_subset, ground_truth_texts) in enumerate(loader):
                        eeg_data_subset, labels_subset, labels_class_subset = eeg_data_subset.to(device), labels_subset.to(device), labels_class_subset.to(device)
                        outputs_subset, class_outputs_subset = model(eeg_data_subset)
                        loss_subset_rec = loss_fn_reconstruction(outputs_subset, labels_subset)
                        loss_subset_class = loss_fn_classification(class_outputs_subset, labels_class_subset)
                        loss_subset = loss_subset_rec + alpha * loss_subset_class

                        _, predicted_test = torch.max(class_outputs_subset.data, 1)
                        total+= labels_class_subset.size(0)
                        correct+= (predicted_test == labels_class_subset).sum().item()

                        _, top5_pred = torch.topk(class_outputs_subset, 5, dim=1)
                        top5_correct += sum([labels_class_subset[i] in top5_pred[i] for i in range(labels_class_subset.size(0))])

                        subset_loss += loss_subset.item()
                        subset_loss_rec += loss_subset_rec.item()
                        subset_loss_class += loss_subset_class.item()

                        all_outputs.append(outputs_subset.cpu().numpy())
                        all_labels.append(labels_subset.cpu().numpy())
                        #logging.info(f"{loss_subset.item()}")

                        # Last batch processing for WAV conversion
                        if j == len(loader) - 2:
                            count = 0
                            for output, label, gt_text, label_class, pred_class in zip(outputs_subset.cpu().numpy(), labels_subset.cpu().numpy(), ground_truth_texts, labels_class_subset.cpu().numpy(), predicted_test.cpu().numpy()):
                                count += 1
                                average_correlation = correlation_persample_nobatch(output, label)
                                gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_gd.wav")
                                pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{pred_class}_{average_correlation:.4f}_pred.wav")
                                mcd = 0
                                if hyperparameters['n_classes'] == 80:
                                    wav_gd = utils.mel_to_wav(label, num_bins=hyperparameters['n_classes'])  # Ground truth
                                    wav_pred = utils.mel_to_wav(output, num_bins=hyperparameters['n_classes'])  # Predictions
                                    mcd = utils.compute_mcd(wav_gd, wav_pred)
                                    mcds[data_subset].append(mcd)
                                    gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_{mcd:.4f}_gd.wav")
                                    pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{pred_class}_{average_correlation:.4f}_{mcd:.4f}_pred.wav")
                                    utils.save_wav(wav_gd, gd_path, sr=16000)
                                    utils.save_wav(wav_pred, pred_path, sr=16000)

                                np.save(gd_path.replace('.wav', '.npy'), label)
                                np.save(pred_path.replace('.wav', '.npy'), output)
                                logging.info(f"correlation {epoch+1}: {data_subset} for {count}: {average_correlation:.4f}, mcd: {mcd:.4f}")


                avg_loss = subset_loss / len(loader)
                avg_loss_rec = subset_loss_rec / len(loader)
                avg_loss_class = subset_loss_class / len(loader)
                acc = correct/total
                top5_acc = top5_correct / total

                #logging.info(f"correlation: {correlations[data_subset]}")
                correlation_p = correlation_persample(all_outputs, all_labels)
                correlation_c = correlation_conca(all_outputs, all_labels)
                correlation_p_dtw = correlation_persample_dtw(all_outputs, all_labels)

                if data_subset == "val":
                    correlation_c_dtw = correlation_conca_dtw(all_outputs, all_labels)
                    correlation_c_dtw = 0

                else:
                    correlation_c_dtw = 0
                correlations[data_subset].append(correlation_c)

                avg_mcd = np.mean(mcds[data_subset])
                logging.info(f"{data_subset.capitalize()} at Epoch {epoch+1}/Step {global_step}: Loss: {avg_loss:.4f}, loss_rec :{avg_loss_rec}, loss_class :{avg_loss_class}, correlation_p :{correlation_p:.4f}, correlation_c:{correlation_c:.4f},mcd : {avg_mcd:.4f}, correlation_p_dtw :{correlation_p_dtw:.4f}, correlation_c_dtw :{correlation_c_dtw:.4f}, acc: {acc:.4f}, top5acc: {top5_acc:.4f}")

                if data_subset == "val":
                    val_loss = avg_loss  # Specific handling for validation loss

            #Save checkpoint including metrics

            checkpoint_path = utils.setup_model_saving(base_dir, hyperparameters_string, current_time, global_step)

            torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'train_loss': avg_loss,  # Assuming avg_loss here is the last computed, i.e., for validation
               'val_loss': val_loss,
               'train_pearson': np.mean(correlations['train']),
               'val_pearson': np.mean(correlations['val']),
            }, checkpoint_path)


            logging.info(f"Checkpoint saved to {checkpoint_path}.")
            logging.info(f"__________________________________")
            model.train() # Switch back to training mode


        # Evaluation within epoch
        if ((epoch+1) % hyperparameters['epochs_within_evaluation'] == 0) and (i == len(train_loader)-1):
            val_check = True
            model.eval()
            val_loss = 0.0
            val_loss_rec = 0.0
            val_loss_class = 0.0
            total = 0
            correct = 0
            top5_correct = 0
            with torch.no_grad():
                all_outputs = []
                all_labels = []
                for eeg_data_test, labels_test, labels_class_test, ground_truth_texts in test_loader:
                    eeg_data_test, labels_test, labels_class_test = eeg_data_test.to(device), labels_test.to(device), labels_class_test.to(device)
                    outputs_test, class_outputs_test = model(eeg_data_test)
                    loss_ev_rec = loss_fn_reconstruction(outputs_test, labels_test)
                    loss_ev_class = loss_fn_classification(class_outputs_test, labels_class_test)
                    loss_ev = loss_ev_rec + alpha * loss_ev_class

                    _, predicted_test = torch.max(class_outputs_test.data, 1)
                    total += labels_class_test.size(0)
                    correct += (predicted_test == labels_class_test).sum().item()

                    # Top-5 accuracy
                    _, top5_pred = torch.topk(class_outputs_test, 5, dim=1)
                    top5_correct += sum([labels_class_test[i] in top5_pred[i] for i in range(labels_class_test.size(0))])

                    val_loss += loss_ev.item()
                    val_loss_rec += loss_ev_rec.item()
                    val_loss_class += loss_ev_class.item()
                    all_outputs.append(outputs_test.cpu().numpy())
                    all_labels.append(labels_test.cpu().numpy())

            val_loss /= len(test_loader)
            val_loss_rec /= len(test_loader)
            val_loss_class /= len(test_loader)
            val_acc = correct / total
            val_top5_acc = top5_correct / total


            val_p = correlation_persample(all_outputs, all_labels)
            val_c = correlation_conca(all_outputs, all_labels)

            logging.info(f"Intermediate Validation Loss at step {global_step}: {val_loss:.4f}, val_loss_rec: {val_loss_rec:.4f}, val_loss_class: {val_loss_class:.4f}, val_acc: {val_acc:.4f}, val_top5_acc: {val_top5_acc:.4f}, val_pcc_p: {val_p:.4f}, val_pcc_c: {val_c:.4f}")
            logging.info(f"__________________________________")

            model.train()  # Switch back to training mode


    scheduler.step() # changed scheduler as epoch steps
    # Print epoch summary
    epoch_loss = running_loss / len(train_loader)
    epoch_loss_reconstruction = running_loss_reconstruction / len(train_loader)
    epoch_loss_classification = running_loss_classification / len(train_loader)
    epoch_end_time = time.time()

    last_time = epoch_end_time - training_start_time
    average_epoch_time = last_time / (epoch + 1)
    remaining_epochs = hyperparameters['num_epochs'] - (epoch + 1)
    estimated_time_left_training = average_epoch_time * remaining_epochs
    all_time = estimated_time_left_training
    last_time = timedelta(seconds=last_time)
    all_time = timedelta(seconds=all_time)



    last_time_formatted = f"{int(last_time.total_seconds()) // 3600:02d}:{int(last_time.total_seconds()) % 3600 // 60:02d}:{int(last_time.total_seconds()) % 60:02d}"
    all_time_formatted = f"{int(all_time.total_seconds()) // 3600:02d}:{int(all_time.total_seconds()) % 3600 // 60:02d}:{int(all_time.total_seconds()) % 60:02d}"


    logging.info(f"{current_time}, Epoch [{epoch+1}/{hyperparameters['num_epochs']}], step: {global_step}, {last_time_formatted}/{all_time_formatted}, Average Loss: {epoch_loss:.4f}, Average Reconstruction Loss: {epoch_loss_reconstruction:.4f}, Average Classification Loss: {epoch_loss_classification:.4f}")

    if val_check:
        if val_loss < best_loss:
            logging.info("Validation loss decreased. Saving the model.")
            logging.info(f"Best model at step {global_step} and epoch {epoch+1}")
            torch.save(model.state_dict(), model_save_path)
            best_loss = val_loss


    if val_check and hyperparameters['patience'] != -1:
        if val_loss < best_loss:
            logging.info("Validation loss decreased. Saving the model.")
            torch.save(model.state_dict(), model_save_path)
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            logging.info(f"EarlyStopping counter: {counter} out of {hyperparameters['patience']}")

        if counter >= hyperparameters['patience']:
            logging.info("Early stopping")
            break


training_end_time = time.time()
logging.info(f"Total training time: {training_end_time - training_start_time:.2f} seconds")



# final evaluate
# final evaluate

logging.info(f"___________ FOR_BEST______________________")
model.load_state_dict(torch.load(model_save_path))
output_path = os.path.join(base_dir, 'best_output')
os.makedirs(output_path, exist_ok=True)

hash_string = utils.get_md5_hash(hyperparameters_string)
hash_string = str(current_time) + "_" + hash_string
out_subdir = os.path.join(output_path, hash_string)
if not os.path.exists(out_subdir):
    os.makedirs(out_subdir)

model.eval()
val_loss = 0.0
val_loss_rec = 0.0
val_loss_class = 0.0
correlations = {'train': [], 'val': []}  # To store Pearson correlations
mcds = {'train': [], 'val': []}
accuracies = {'train': [], 'val': []}
top5_accuracies = {'train': [], 'val': []}

# Unified handling for both datasets
for data_subset, loader in [("train", train_loader), ("val", test_loader)]:
    subset_loss = 0.0
    subset_loss_rec = 0.0
    subset_loss_class = 0.0
    total = 0
    correct = 0
    top5_correct = 0

    with torch.no_grad():
        count = 0
        all_outputs = []
        all_labels = []
        for j, (eeg_data_subset, labels_subset, labels_class_subset, ground_truth_texts) in enumerate(loader):
            eeg_data_subset, labels_subset, labels_class_subset = eeg_data_subset.to(device), labels_subset.to(device), labels_class_subset.to(device)
            outputs_subset, class_outputs_subset = model(eeg_data_subset)
            loss_subset_rec = loss_fn_reconstruction(outputs_subset, labels_subset)
            loss_subset_class = loss_fn_classification(class_outputs_subset, labels_class_subset)
            loss_subset = loss_subset_rec + alpha * loss_subset_class

            _, predicted_test = torch.max(class_outputs_subset.data, 1)
            total += labels_class_subset.size(0)
            correct += (predicted_test == labels_class_subset).sum().item()

            # Top-5 accuracy
            _, top5_pred = torch.topk(class_outputs_subset, 5, dim=1)
            top5_correct += sum([labels_class_subset[i] in top5_pred[i] for i in range(labels_class_subset.size(0))])

            subset_loss += loss_subset.item()
            subset_loss_rec += loss_subset_rec.item()
            subset_loss_class += loss_subset_class.item()
            all_outputs.append(outputs_subset.cpu().numpy())
            all_labels.append(labels_subset.cpu().numpy())

            for output, label, gt_text, label_class in zip(outputs_subset.cpu().numpy(), labels_subset.cpu().numpy(), ground_truth_texts, labels_class_subset.cpu().numpy()):
                count += 1

                average_correlation = correlation_persample_nobatch(output, label)
                gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_gd.wav")
                pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_pred.wav")
                mcd = 0
                if hyperparameters['n_classes'] == 80:
                    wav_gd = utils.mel_to_wav(label, num_bins=hyperparameters['n_classes'])  # Ground truth
                    wav_pred = utils.mel_to_wav(output, num_bins=hyperparameters['n_classes'])  # Predictions
                    mcd = utils.compute_mcd(wav_gd, wav_pred)
                    mcds[data_subset].append(mcd)
                    gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_{mcd:.4f}_gd.wav")
                    pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_{mcd:.4f}_pred.wav")
                    utils.save_wav(wav_gd, gd_path, sr=16000)
                    utils.save_wav(wav_pred, pred_path, sr=16000)

                np.save(gd_path.replace('.wav', '.npy'), label)
                np.save(pred_path.replace('.wav', '.npy'), output)
                logging.info(f"correlation {epoch+1}: {data_subset} for {count}:{average_correlation:.4f}, mcd: {mcd:.4f}")

    # Logging and saving after processing each dataset
    avg_loss = subset_loss / len(loader)
    avg_loss_rec = subset_loss_rec / len(loader)
    avg_loss_class = subset_loss_class / len(loader)
    acc = correct / total
    top5_acc = top5_correct / total
    correlation_p = correlation_persample(all_outputs, all_labels)
    correlation_c = correlation_conca(all_outputs, all_labels)
    correlation_p_dtw = correlation_persample_dtw(all_outputs, all_labels)
    if data_subset == "val":
        correlation_c_dtw = correlation_conca_dtw(all_outputs, all_labels)
    else:
        correlation_c_dtw = 0
    avg_mcd = np.mean(mcds[data_subset])
    accuracies[data_subset].append(acc)
    top5_accuracies[data_subset].append(top5_acc)
    logging.info(f"{data_subset.capitalize()} at Epoch {epoch+1}/Step {global_step}: Loss: {avg_loss:.4f}, loss_rec: {avg_loss_rec:.4f}, loss_class: {avg_loss_class:.4f}, acc: {acc:.4f}, top5_acc: {top5_acc:.4f}, correlation_p: {correlation_p:.4f}, correlation_c: {correlation_c:.4f}, mcd: {avg_mcd:.4f}, correlation_p_dtw: {correlation_p_dtw:.4f}, correlation_c_dtw: {correlation_c_dtw:.4f}")

logging.info(f"_____________END_BEST___________________")

logging.info(f"___________ FOR_LAST______________________")
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])

output_path = os.path.join(base_dir, 'last_output')
os.makedirs(output_path, exist_ok=True)

hash_string = utils.get_md5_hash(hyperparameters_string)
hash_string = str(current_time) + "_" + hash_string
out_subdir = os.path.join(output_path, hash_string)
if not os.path.exists(out_subdir):
    os.makedirs(out_subdir)

model.eval()
val_loss = 0.0
val_loss_rec = 0.0
val_loss_class = 0.0
correlations = {'train': [], 'val': []}  # To store Pearson correlations
mcds = {'train': [], 'val': []}
accuracies = {'train': [], 'val': []}
top5_accuracies = {'train': [], 'val': []}

for data_subset, loader in [("train", train_loader), ("val", test_loader)]:
    subset_loss = 0.0
    subset_loss_rec = 0.0
    subset_loss_class = 0.0
    total = 0
    correct = 0
    top5_correct = 0

    with torch.no_grad():
        count = 0
        all_outputs = []
        all_labels = []
        for j, (eeg_data_subset, labels_subset, labels_class_subset, ground_truth_texts) in enumerate(loader):
            eeg_data_subset, labels_subset, labels_class_subset = eeg_data_subset.to(device), labels_subset.to(device), labels_class_subset.to(device)
            outputs_subset, class_outputs_subset = model(eeg_data_subset)
            loss_subset_rec = loss_fn_reconstruction(outputs_subset, labels_subset)
            loss_subset_class = loss_fn_classification(class_outputs_subset, labels_class_subset)
            loss_subset = loss_subset_rec + alpha * loss_subset_class

            _, predicted_test = torch.max(class_outputs_subset.data, 1)
            total += labels_class_subset.size(0)
            correct += (predicted_test == labels_class_subset).sum().item()

            # Top-5 accuracy
            _, top5_pred = torch.topk(class_outputs_subset, 5, dim=1)
            top5_correct += sum([labels_class_subset[i] in top5_pred[i] for i in range(labels_class_subset.size(0))])

            subset_loss += loss_subset.item()
            subset_loss_rec += loss_subset_rec.item()
            subset_loss_class += loss_subset_class.item()
            all_outputs.append(outputs_subset.cpu().numpy())
            all_labels.append(labels_subset.cpu().numpy())

            for output, label, gt_text, label_class in zip(outputs_subset.cpu().numpy(), labels_subset.cpu().numpy(), ground_truth_texts, labels_class_subset.cpu().numpy()):
                count += 1

                average_correlation = correlation_persample_nobatch(output, label)
                gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_gd.wav")
                pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_pred.wav")
                mcd = 0
                if hyperparameters['n_classes'] == 80:
                    wav_gd = utils.mel_to_wav(label, num_bins=hyperparameters['n_classes'])  # Ground truth
                    wav_pred = utils.mel_to_wav(output, num_bins=hyperparameters['n_classes'])  # Predictions
                    mcd = utils.compute_mcd(wav_gd, wav_pred)
                    mcds[data_subset].append(mcd)
                    gd_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_{mcd:.4f}_gd.wav")
                    pred_path = os.path.join(out_subdir, f"{global_step}_{data_subset}_{count}_{gt_text}_{average_correlation:.4f}_{mcd:.4f}_pred.wav")
                    utils.save_wav(wav_gd, gd_path, sr=16000)
                    utils.save_wav(wav_pred, pred_path, sr=16000)

                np.save(gd_path.replace('.wav', '.npy'), label)
                np.save(pred_path.replace('.wav', '.npy'), output)
                logging.info(f"correlation {epoch+1}: {data_subset} for {count}:{average_correlation:.4f}, mcd: {mcd:.4f}")

    # Logging and saving after processing each dataset
    avg_loss = subset_loss / len(loader)
    avg_loss_rec = subset_loss_rec / len(loader)
    avg_loss_class = subset_loss_class / len(loader)
    acc = correct / total
    top5_acc = top5_correct / total
    correlation_p = correlation_persample(all_outputs, all_labels)
    correlation_c = correlation_conca(all_outputs, all_labels)
    correlation_p_dtw = correlation_persample_dtw(all_outputs, all_labels)
    if data_subset == "val":
        correlation_c_dtw = correlation_conca_dtw(all_outputs, all_labels)
    else:
        correlation_c_dtw = 0
    avg_mcd = np.mean(mcds[data_subset])
    accuracies[data_subset].append(acc)
    top5_accuracies[data_subset].append(top5_acc)
    logging.info(f"{data_subset.capitalize()} at Epoch {epoch+1}/Step {global_step}: Loss: {avg_loss:.4f}, loss_rec: {avg_loss_rec:.4f}, loss_class: {avg_loss_class:.4f}, acc: {acc:.4f}, top5_acc: {top5_acc:.4f}, correlation_p: {correlation_p:.4f}, correlation_c: {correlation_c:.4f}, mcd: {avg_mcd:.4f}, correlation_p_dtw: {correlation_p_dtw:.4f}, correlation_c_dtw: {correlation_c_dtw:.4f}")

logging.info(f"____________END_LAST_____________________")
