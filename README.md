# 🧬 Histopathology OOD Classification — MVA DLMI 2025 Kaggle Challenge

> Official Kaggle challenge: [https://www.kaggle.com/competitions/mva-dlmi-2025-histopathology-ood-classification](https://www.kaggle.com/competitions/mva-dlmi-2025-histopathology-ood-classification)

---

## 🧠 Overview

This project addresses the binary classification of histopathology image patches extracted from whole slide images (WSIs). The key challenge is to generalize under **domain shift** — training, validation, and test data come from **different hospitals** with varying staining protocols and imaging equipment.

The goal is to build a robust classifier that performs well across centers using pretrained visual encoders and carefully tuned training pipelines.

---

## 📁 Project Structure

```bash
.
├── classifiers.py              # MLP, SEHead, and classifier selector
├── datasets.py                 # PyTorch Datasets for raw and precomputed data
├── feature_extractor.py       # Feature extractor loader (ProVGigaPath, KimiaNet, CTransPath)
├── main.py                    # Entry point — runs the full training and test pipeline
├── models.py                  # ConvStem and Swin Transformer (CTransPath logic)
├── pipeline.py                # Training, validation, and test loop logic
├── preprocessing.py           # Transformations and feature precomputation
├── utils.py                   # Utility function to precompute features
├── timm-0.5.4.tar             # Custom TIMM version for CTransPath
├── pretrained_models/         # Folder for downloaded model weights
├── output_files/              # Folder where submission CSV is saved
└── data/                      # HDF5 files: train.h5, val.h5, test.h5
```

---

## 🔧 Setup Instructions

### 1. 📦 Install Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

> 💡 **If using the `ctranspath` model:**

You must uninstall the standard `timm` library and install the specific version provided with the repo:

```bash
pip uninstall timm
pip install timm-0.5.4.tar
```

This file is downloaded from the original CTransPath repository:  
🔗 https://github.com/Xiyue-Wang/TransPath

or by following the link: 
🔗 https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view

---

### 2. 🔐 Hugging Face Login for ProVGigaPath

If you're using the `provgigapath` feature extractor, set your Hugging Face token in the environment:

```bash
export HF_TOKEN=your_hf_token_here
```

You can get the token from your Hugging Face account settings.  
Model link on Hugging Face:  
🔗 https://huggingface.co/prov-gigapath/prov-gigapath

---

### 3. 🔽 Download Pretrained Weights

You must download the following model weights for feature extractors:

#### 🧠 CTransPath
- Download `ctranspath.pth` from either:
  - The [official GitHub repository](https://github.com/Xiyue-Wang/TransPath)
  - Or directly from this Google Drive link:  
    🔗 https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view  
- Place the file in a folder named:

```bash
pretrained_models/ctranspath.pth
```

#### 🔬 KimiaNet
- Download `KimiaNetPyTorchWeights.pth` from the official GitHub:  
  🔗 https://github.com/KimiaLabMayo/KimiaNet  
- Place the file in:

```bash
pretrained_models/KimiaNetPyTorchWeights.pth
```

---

## 🏁 Running the Pipeline

To run the training and prediction pipeline, simply execute:

```bash
python main.py
```

This will:

- ✅ Load the selected feature extractor (`provgigapath`, `kimianet`, or `ctranspath`)
- 🔁 Preprocess and augment the image patches
- ⚙️ Precompute features from the backbone model
- 🧠 Train the classifier using early stopping
- 💾 Save the best model based on validation loss
- 📤 Generate the final CSV for Kaggle submission

---

## 🧪 Feature Extractors Supported

| Name         | Description                                     | Source                                                      |
|--------------|-------------------------------------------------|-------------------------------------------------------------|
| `provgigapath` | SOTA transformer trained on gigapixel pathology data | [HuggingFace](https://huggingface.co/prov-gigapath/prov-gigapath) |
| `kimianet`     | DenseNet121 pretrained on pathology slides     | [KimiaNet GitHub](https://github.com/KimiaLabMayo/KimiaNet) |
| `ctranspath`   | Swin Transformer + ConvStem architecture       | [CTransPath GitHub](https://github.com/Xiyue-Wang/TransPath) |

---

## 🧪 Preprocessing & Augmentation

### ✅ During training, the pipeline applies:

- `RandomResizedCrop`
- `ColorJitter` (brightness, contrast, saturation, hue)
- `RandomAffine` transformations
- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `Normalize` (ImageNet statistics)

### 🧪 During validation and testing:

- Resize to a fixed size
- Center crop
- Normalize (ImageNet statistics)

This approach improves robustness to variations in staining, magnification, and acquisition conditions across centers.

---

## 📦 Dataset

The dataset used in this challenge can be downloaded directly from the official Kaggle competition page:  
🔗 https://www.kaggle.com/competitions/mva-dlmi-2025-histopathology-ood-classification

Make sure to download and place the files (`train.h5`, `val.h5`, `test.h5`) into the `data/` folder.

---

## 📤 Output Format

After training and inference, a submission file is saved to:

```bash
output_files/baseline_<FEATURE_NAME>.csv
```

The format must match the required Kaggle submission format:

```csv
ID,Pred
0,0
1,1
2,0
...
```

---

## 💡 Tips

- 🧪 For better generalization, enable **strong data augmentation** (already included in the training transform).
- ✅ Use a dedicated classifier for each feature extractor, chosen automatically in `classifiers.py`.

---

## 📚 References

- [Kaggle Challenge Page](https://www.kaggle.com/competitions/mva-dlmi-2025-histopathology-ood-classification)
- [CTransPath Paper & GitHub](https://github.com/Xiyue-Wang/TransPath)
- [KimiaNet Paper & GitHub](https://github.com/KimiaLabMayo/KimiaNet)
- [ProVGigaPath Model on HuggingFace](https://huggingface.co/prov-gigapath/prov-gigapath)
