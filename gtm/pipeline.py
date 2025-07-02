import os
import pickle
from pprint import pprint
import time
from typing import Dict
import shap
import lightgbm as lgb
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from gtm.dataset import GeoPretrainDataset, GeoTaskDataset, GeoDoubleAngleDataset
from gtm.model import LitGeoPretrainModel, LitGeoTaskModel, LitGeoDoubleAngleModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score



def run_pretrain_pipeline(task_config: Dict):
    pprint(task_config)

    if not os.path.exists(task_config["output_folder_path"]):
        os.makedirs(task_config["output_folder_path"])

    if task_config["pretrain_data_path"].endswith('csv'):
        csv_data = pd.read_csv(task_config["pretrain_data_path"])
    elif task_config["pretrain_data_path"].endswith('feather'):
        csv_data = pd.read_feather(task_config["pretrain_data_path"])
    else:
        csv_data = pd.read_pickle(task_config["pretrain_data_path"])

    standard_scaler = StandardScaler()
    csv_data[task_config["dataset"]["input_cont_cols"]] = \
        standard_scaler.fit_transform(csv_data[task_config["dataset"]["input_cont_cols"]])
    save_path = os.path.join(task_config["output_folder_path"], "input_cont_scaler.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(standard_scaler, f)

    standard_scaler = StandardScaler()
    csv_data[task_config["dataset"]["space_target_cols"]] = \
        standard_scaler.fit_transform(csv_data[task_config["dataset"]["space_target_cols"]])
    if task_config["dataset"]["time_target_cols"]:
        standard_scaler = StandardScaler()
        csv_data[task_config["dataset"]["time_target_cols"]] = \
            standard_scaler.fit_transform(csv_data[task_config["dataset"]["time_target_cols"]])

    if task_config["dataset"]["input_cate_cols"]:
        label_encoders = {}
        for c in task_config["dataset"]["input_cate_cols"]:
            label_encoder = LabelEncoder()
            csv_data[c] = label_encoder.fit_transform(csv_data[c])
            label_encoders[c] = label_encoder
        save_path = os.path.join(task_config["output_folder_path"], "input_cate_encoders.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(label_encoders, f)

    train_data, valid_data = train_test_split(csv_data, test_size=0.1, random_state=task_config["seed"])
    train_set = GeoPretrainDataset(train_data, **task_config["dataset"])
    valid_set = GeoPretrainDataset(valid_data, **task_config["dataset"])
    train_loader = DataLoader(train_set, shuffle=True, **task_config["dataloader"])
    valid_loader = DataLoader(valid_set, shuffle=False, **task_config["dataloader"])

    model = LitGeoPretrainModel(
        n_feats=train_set.num_cont_feats,
        n_cates=train_set.num_cate_types,
        n_space_outs=train_set.num_space_outs,
        n_time_outs=train_set.num_time_outs,
        **task_config["model"]
    )

    model_checkpoint = ModelCheckpoint(**task_config["callbacks"]["model_checkpoint"])
    early_stopping = EarlyStopping(**task_config["callbacks"]["early_stopping"])

    trainer = pl.Trainer(
        default_root_dir=task_config["output_folder_path"],
        callbacks=[model_checkpoint, early_stopping],
        **task_config["trainer"]
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint(os.path.join(task_config["output_folder_path"], "pretrain_model.ckpt"))

def run_finetune_pipeline(task_config: Dict):
    pprint(task_config)

    train_data = pd.read_csv(task_config["train_data_path"])
    valid_data = pd.read_csv(task_config["valid_data_path"])

    if os.path.isdir(task_config["test_data_path"]):
        test_data_path = os.listdir(task_config["test_data_path"])
        test_data_path = [os.path.join(task_config["test_data_path"], p) for p in test_data_path]
    else:
        test_data_path = [task_config["test_data_path"]]

    with open(task_config["scaler_path"], "rb") as f:
        standard_scaler = pickle.load(f)
    for input_data in [train_data, valid_data]:
        input_data[task_config["dataset"]["input_cont_cols"]] = \
            standard_scaler.transform(input_data[task_config["dataset"]["input_cont_cols"]])

    if task_config["dataset"]["input_cate_cols"]:
        with open(task_config["encoder_path"], "rb") as f:
            label_encoders = pickle.load(f)
        for input_data in [train_data, valid_data] :
            for c, label_encoder in label_encoders.items():
                input_data[c] = label_encoder.transform(input_data[c])

    train_set = GeoTaskDataset(train_data, **task_config["dataset"])
    valid_set = GeoTaskDataset(valid_data, **task_config["dataset"])

    train_loader = DataLoader(train_set, shuffle=True, **task_config["dataloader"])
    valid_loader = DataLoader(valid_set, shuffle=False, **task_config["dataloader"])

    model = LitGeoTaskModel(n_task_out=train_set.num_task_outs, **task_config["model"])

    model_checkpoint = ModelCheckpoint(**task_config["callbacks"]["model_checkpoint"])
    early_stopping = EarlyStopping(**task_config["callbacks"]["early_stopping"])

    trainer = pl.Trainer(
        default_root_dir=task_config["output_folder_path"],
        callbacks=[model_checkpoint, early_stopping],
        **task_config["trainer"]
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.save_checkpoint(os.path.join(task_config["output_folder_path"], "task_model.ckpt"))

    for i, data_path in enumerate(test_data_path):
        test_data = pd.read_csv(data_path)

        test_data[task_config["dataset"]["input_cont_cols"]] = \
            standard_scaler.transform(test_data[task_config["dataset"]["input_cont_cols"]])
        
        if task_config["dataset"]["input_cate_cols"]:
            for c, label_encoder in label_encoders.items():
                test_data[c] = label_encoder.transform(test_data[c])

        test_name = os.path.basename(test_data_path[i]).split(".")[0]

        test_set = GeoTaskDataset(test_data, **task_config["dataset"])
        test_loader = DataLoader(test_set, shuffle=False, **task_config["dataloader"])

        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            print("Best Checkpoint not Found! Using Current Weights for Prediction ...")
            ckpt_path = None
        predictions_all = trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_path)

        # 拆分出任务输出和中间特征
        task_outputs, features = zip(*predictions_all)
        predictions_task = torch.cat(task_outputs, dim=0).cpu().numpy()
        predictions_features = torch.cat(features, dim=0).cpu().numpy()
        for j in range(len(task_config['dataset']['task_target_cols'])):
            task_name = task_config['dataset']['task_target_cols'][j]
            print(f"Task {task_name}:")
            y_true = test_set.task_targets_data[:, j]
            y_pred = predictions_task[:, j]
            print(f"RMSE: {mean_squared_error(y_true, y_pred) ** 0.5:.3f}")
            print(f"R2: {r2_score(y_true, y_pred):.3f}")
            test_data[f"{task_name}_PRED"] = y_pred
        features_df = pd.DataFrame(
            predictions_features,
            columns=[f"x_feature_{col_idx}" for col_idx in range(predictions_features.shape[1])]
        )
        test_data = pd.concat([test_data, features_df], axis=1)

        # 保存预测结果
        test_data.to_csv(os.path.join(task_config["output_folder_path"], f"{test_name}_pred.csv"), index=False)

      
def run_finetune_inference_pipeline(task_config: Dict):
    pprint(task_config)
    if not os.path.exists(task_config["output_folder_path"]):
        os.makedirs(task_config["output_folder_path"])

    if os.path.isdir(task_config["test_data_path"]):
        test_data_path = os.listdir(task_config["test_data_path"])
        test_data_path = [os.path.join(task_config["test_data_path"], p) for p in test_data_path]
    else:
        test_data_path = [task_config["test_data_path"]]

    model = LitGeoTaskModel.load_from_checkpoint(task_config["model_checkpoint_path"])
    trainer = pl.Trainer(default_root_dir=task_config["output_folder_path"], **task_config["trainer"])

    for i, data_path in enumerate(test_data_path):
        test_data = pd.read_csv(data_path)

        with open(task_config["scaler_path"], "rb") as f:
            standard_scaler = pickle.load(f)
        test_data[task_config["dataset"]["input_cont_cols"]] = \
            standard_scaler.transform(test_data[task_config["dataset"]["input_cont_cols"]])
        
        if task_config["dataset"]["input_cate_cols"]:
            with open(task_config["encoder_path"], "rb") as f:
                label_encoders = pickle.load(f)
            for c, label_encoder in label_encoders.items():
                test_data[c] = label_encoder.transform(test_data[c])

        test_name = os.path.basename(test_data_path[i]).split(".")[0]

        test_set = GeoTaskDataset(test_data, **task_config["dataset"])
        test_loader = DataLoader(test_set, shuffle=False, **task_config["dataloader"])

        predictions_all = trainer.predict(model, dataloaders=test_loader, ckpt_path=task_config["model_checkpoint_path"])
        predictions, features = zip(*predictions_all)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()

        for j in range(len(task_config['dataset']['task_target_cols'])):
            task_name = task_config['dataset']['task_target_cols'][j]
            print(f"Task {task_name}:")
            y_true = test_set.task_targets_data[:, j]
            y_pred = predictions[:, j]
            print(f"RMSE: {mean_squared_error(y_true, y_pred) ** 0.5:.3f}")
            print(f"R2: {r2_score(y_true, y_pred):.3f}")
            print(f"R(Pearson): {pearsonr(y_true, y_pred)[0]:.3f}")
            test_data[f"{task_name}_PRED"] = y_pred
        test_data.to_csv(os.path.join(task_config["output_folder_path"], f"{test_name}_pred.csv"), index=False)


