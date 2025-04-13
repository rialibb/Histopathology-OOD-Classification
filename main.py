from pipeline import pipeline





if __name__ == '__main__':
    
    pipeline(
        FEATURE_NAME = 'provgigapath',                  # str: Feature extractor name ('provgigapath', 'kimianet', 'ctranspath')
        TRAIN_IMAGES_PATH = 'data/train.h5',            # str: Path to training HDF5 file
        VAL_IMAGES_PATH = 'data/val.h5',                # str: Path to validation HDF5 file
        TEST_IMAGES_PATH = 'data/test.h5',              # str: Path to test HDF5 file
        PREPROCESSED_FOLDER = 'preprocessed_data',      # str: Directory to save/load precomputed features
        SEED = 0,                                        # int: Random seed (≥ 0)
        BATCH_SIZE = 64,                                 # int: Batch size for DataLoaders (> 0, e.g., 16–256)
        OPTIMIZER = 'Adam',                              # str: Optimizer name (e.g., 'Adam', 'SGD', 'AdamW')
        OPTIMIZER_PARAMS = {'lr': 0.001},                # dict: Parameters for the optimizer (e.g., learning rate, weight decay)
        LOSS = 'BCELoss',                                # str: Loss function (e.g., 'BCELoss', 'BCEWithLogitsLoss')
        METRIC = 'Accuracy',                             # str: Evaluation metric from torchmetrics (e.g., 'Accuracy', 'AUROC')
        NUM_EPOCHS = 100,                                # int: Maximum number of training epochs (≥ 1)
        PATIENCE = 10,                                   # int: Early stopping patience in epochs (≥ 0)
    )