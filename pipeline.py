import h5py
import torch
import torch.nn as nn
import os
import random
import numpy as np
import pandas as pd
import torchmetrics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import PrecomputedDataset

from preprocessing import preprocess_dataset, preprocessor
from feature_extractor import extractor_model
from classifiers import choose_classifier



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def pipeline(
            FEATURE_NAME = 'provgigapath',
            TRAIN_IMAGES_PATH = 'data/train.h5',
            VAL_IMAGES_PATH = 'data/val.h5',
            TEST_IMAGES_PATH = 'data/test.h5',
            PREPROCESSED_FOLDER = 'preprocessed_data',
            SEED = 0,
            BATCH_SIZE = 64,
            OPTIMIZER = 'Adam',
            OPTIMIZER_PARAMS = {'lr': 0.001},
            LOSS = 'BCELoss',
            METRIC = 'Accuracy',
            NUM_EPOCHS = 100,
            PATIENCE = 10,
):
    """
    Complete training and inference pipeline for binary classification using histopathology features.

    Inputs:
        - FEATURE_NAME (str): Name of the feature extractor to use ('provgigapath', 'kimianet', or 'ctranspath').
        - TRAIN_IMAGES_PATH (str): Path to HDF5 training image file.
        - VAL_IMAGES_PATH (str): Path to HDF5 validation image file.
        - TEST_IMAGES_PATH (str): Path to HDF5 test image file.
        - PREPROCESSED_FOLDER (str): Folder where extracted features are saved/loaded.
        - SEED (int): Random seed for reproducibility.
        - BATCH_SIZE (int): Batch size for training and evaluation.
        - OPTIMIZER (str): Name of the optimizer to use (e.g., 'Adam').
        - OPTIMIZER_PARAMS (dict): Parameters to initialize the optimizer with.
        - LOSS (str): Loss function to use (e.g., 'BCELoss').
        - METRIC (str): Metric to track during training (e.g., 'Accuracy').
        - NUM_EPOCHS (int): Maximum number of training epochs.
        - PATIENCE (int): Number of epochs to wait for improvement before early stopping.

    Outputs:
        - None (saves best model to disk and generates a CSV submission file with test predictions).
    """

    # fix randomness
    torch.random.manual_seed(SEED)
    random.seed(SEED)

    # define preprocessor
    preprocessing_tr, preprocessing_val = preprocessor(FEATURE_NAME)
    
    # define feature extractor
    feature_extractor, FEATURE_DIM = extractor_model(FEATURE_NAME, device)

    # Check if precomputed files exist
    train_file = os.path.join(PREPROCESSED_FOLDER, f'train_features_{FEATURE_NAME}.pt')         
    val_file = os.path.join(PREPROCESSED_FOLDER, f'val_features_{FEATURE_NAME}.pt')             

    if not all(os.path.exists(p) for p in [train_file, val_file]):
        print("Preprocessed data not found. Running preprocessing...")
        preprocess_dataset(preprocessing_tr, preprocessing_val, feature_extractor, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH, PREPROCESSED_FOLDER, train_file, val_file, BATCH_SIZE, SEED, device)
    else:
        print("Preprocessed data found. Skipping preprocessing.")


    # Load precomputed data
    train_data = torch.load(train_file)
    val_data = torch.load(val_file)

    train_dataset = PrecomputedDataset(train_data['features'], train_data['labels'])
    val_dataset = PrecomputedDataset(val_data['features'], val_data['labels'])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # define classifier
    classifier = choose_classifier(FEATURE_NAME)(FEATURE_DIM).to(device)
    
    # define training metrics
    optimizer = getattr(torch.optim, OPTIMIZER)(classifier.parameters(), **OPTIMIZER_PARAMS)
    criterion = getattr(torch.nn, LOSS)()
    metric = getattr(torchmetrics, METRIC)('binary')
    min_loss, best_epoch = float('inf'), 0

    # start training
    print('-'*80)
    print('-'*33+ 'START TRAINING' + '-'*33)
    print('-'*80)
    for epoch in range(NUM_EPOCHS):
        classifier.train()
        train_metrics, train_losses = [], []
        for train_x, train_y in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            train_pred = classifier(train_x.to(device))
            loss = criterion(train_pred, train_y.to(device))
            loss.backward()
            optimizer.step()
            train_losses.extend([loss.item()]*len(train_y))
            train_metric = metric(train_pred.cpu(), train_y.int().cpu())
            train_metrics.extend([train_metric.item()]*len(train_y))
        print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')

        classifier.eval()
        val_metrics, val_losses = [], []
        for val_x, val_y in tqdm(val_dataloader, leave=False):
            with torch.no_grad():
                val_pred = classifier(val_x.to(device))
            loss = criterion(val_pred, val_y.to(device))
            val_losses.extend([loss.item()]*len(val_y))
            val_metric = metric(val_pred.cpu(), val_y.int().cpu())
            val_metrics.extend([val_metric.item()]*len(val_y))
        print(f'Epoch valid [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(val_losses):.4f} | Metric {np.mean(val_metrics):.4f}')

        if np.mean(val_losses) < min_loss:
            mean_val_loss = np.mean(val_losses)
            print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
            min_loss = mean_val_loss
            best_epoch = epoch
            # create pretrained classifiers file:
            if not os.path.exists('pretrained_classifiers'):
                os.makedirs('pretrained_classifiers')
        
            torch.save(classifier.state_dict(), f'pretrained_classifiers/best_model_{FEATURE_NAME}.pth')                

        if epoch - best_epoch == PATIENCE:
            break
        

    # start testing
    print('-'*80)
    print('-'*32+ 'START TESTING' + '-'*32)
    print('-'*80)

    classifier.load_state_dict(torch.load(f'pretrained_classifiers/best_model_{FEATURE_NAME}.pth', weights_only=True))   
    classifier.eval()
    classifier.to(device)


    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        test_ids = list(hdf.keys())

    solutions_data = {'ID': [], 'Pred': []}
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        for test_id in tqdm(test_ids):
            img = preprocessing_val(torch.tensor(np.array(hdf.get(test_id).get('img')))).unsqueeze(0).float()
            pred = classifier(feature_extractor(img.to(device))).detach().cpu()
            solutions_data['ID'].append(int(test_id))
            solutions_data['Pred'].append(int(pred.item() > 0.5))
    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    # create submission file:
    if not os.path.exists('output_files'):
        os.makedirs('output_files')
    # save output
    solutions_data.to_csv(f'output_files/baseline_{FEATURE_NAME}.csv')                                               