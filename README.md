**<h1>PIST-DNN: A Two-Stage Deep Learning Framework for Pixel-wise Aerosol Inversion</h1>** 
This tutorial provides step-by-step guidance on how to use **PIST-DNN**, a two-stage deep learning framework designed for satellite-based aerosol retrieval. The framework includes:
* **Pretraining** using pixel-wise data without ground truth
* **Finetuning** using data with real station observations

All input datasets are allowed to contain missing values > (NaN). The code is structured to handle this seamlessly.

---
**<h2>📁 Overview of File Structure</h2>**
```
 ├── config/                        # YAML configs for each stage
 │   ├── pretrain.yaml              # config for pretraining
 │   ├── finetune.yaml              # config for finetuning
 │   └── finetune_inference.yaml    # config for inference
 ├── data/
 │   ├── pretrain/                  # input data for pretraining
 │   └── train/                     # input data for finetuning
 ├── gtm/                           
 │   ├── config.py
 │   ├── dataset.py
 │   ├── model.py
 │   └── pipeline.py
 ├── outputs/                      
 ├── main.py                       
 ├── environment.yml                # Python environment config
 └── README.md
```

---
**<h2>🛠️ Environment Setup</h2>**
```
conda env create -f environment.yml
conda activate pist-dnn
```

---
**<h2>1️⃣ Step 1: Pretraining (Unsupervised)</h2>**
The pretraining stage trains the model using only pixel-level features, without requiring station observations. This helps the model learn general spatial-temporal representations.   
**<h3>Data Requirements</h3>**  
* **Format:** ```.csv```, ```.feather```, or ```.pkl```
* **File:** placed under ```data/pretrain/pretrain_data.feather```
* **Features:**
  * ```input_cont_cols```: continuous variables for the current pixel
  * ```input_cate_cols```: categorical variables for the current pixel
  * ```space_target_cols```: features from surrounding 8 pixels (spatial context)
  * ```time_target_cols```: features from adjacent days (temporal context)  

**<h3>Config Setup:</h3>** ```config/pretrain.yaml```
```
pretrain_data_path: "data/pretrain/pretrain_data.feather"
output_folder_path: "outputs/pretrain"

input_cont_cols: [...continuous variables...]
input_cate_cols: [...categorical variables...]
space_target_cols: [... spatial features ...]
time_target_cols: [... temporal features ...]

max_epochs:      # maximum number of training epochs
patience:        # number of epochs to wait before early stopping
```
**<h3>Run Pretraining</h3>**
```python main.py -t pretrain -c config/pretrain.yaml```
**<h3>Output</h3>**
Checkpoint will be saved to:  
```outputs/pretrain/lightning_logs/version_X/checkpoints/epoch=XX-step=XXXXX.ckpt```

---
**<h2>2️⃣ Step 2: Finetuning (Supervised)</h2>**
The finetuning stage refines the pretrained model using ground-truth station data. This is a supervised training phase.
**<h3>Data Requirements</h3>**
* **Format:** ```.csv```, placed under ```data/train/```，eg.:
  * faod+caod_2001-2024_train_beijing.csv
  * faod+caod_2001-2024_valid_beijing.csv
  * faod+caod_2001-2024_test_beijing.csv
* **Features:**
  * ```input_cont_cols:``` continuous features of the current pixel
  * ```input_cate_cols:``` categorical features (e.g., month)
  * ```task_target_cols:``` target values (e.g., faod, caod)
  * 
**<h3>Config Setup:</h3>** ```config/finetune.yaml```
```
train_data_path: data/train/faod+caod_2001-2024_train_beijing.csv
valid_data_path: data/train/faod+caod_2001-2024_valid_beijing.csv
test_data_path: data/train/faod+caod_2001-2024_test_beijing.csv

scaler_path: outputs/pretrain/input_cont_scaler.pkl
encoder_path: outputs/pretrain/input_cate_encoders.pkl

pretrain_ckpt_path: outputs/pretrain/lightning_logs/version_X/checkpoints/epoch=XX-step=XXXXX.ckpt

input_cont_cols: [...continuous variables...]
input_cate_cols: [...categorical variables...]
task_target_cols: [...target variables...]

max_epochs:      # maximum number of training epochs
patience:        # number of epochs to wait before early stopping
```
**<h3>Run Finetuning</h3>**
```python main.py -t finetune -c config/finetune.yaml```
**<h3>Output</h3>**
* Predictions on test set
* 128 ```x_feature_*``` vectors saved with test results

---
**<h2>3️⃣ Step 3: Visualize Model Outputs</h2>**
After testing, the output CSV will contain 128 ```x_feature_*``` columns. You can apply dimension reduction techniques to visualize the learned feature space and assess separation by season or region:  
* t-SNE
* PCA
* UMAP

**<h2>📝 Citation</h2>**
If you use this framework, please cite or acknowledge the corresponding paper.
