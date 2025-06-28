**PIST-DNN: A Two-Stage Deep Learning Framework for Pixel-wise Aerosol Inversion**
This tutorial provides step-by-step guidance on how to use **PIST-DNN**, a two-stage deep learning framework designed for satellite-based aerosol retrieval. The framework includes:
* **Pretraining** using pixel-wise data without ground truth
* **Finetuning** using data with real station observations
All input datasets are allowed to contain missing values (NaN). The code is structured to handle this seamlessly.
---
**Overview of File Structure**
`├── config/                         # YAML configs for each stage
 │   ├── pretrain.yaml              # config for pretraining
 │   ├── finetune.yaml              # config for finetuning
 │   └── finetune_inference.yaml    # config for inference
 ├── data/
 │   ├── pretrain/                  # input data for pretraining (no ground truth)
 │   └── train/                     # input data for finetuning (with ground truth)
 ├── gtm/                           # model and data logic
 │   ├── config.py
 │   ├── dataset.py
 │   ├── model.py
 │   └── pipeline.py
 ├── outputs/                       # model outputs and checkpoints
 ├── main.py                        # main training script
 ├── environment.yml                # Python environment config
 └── README.md `
