from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import R2Score

from gtm.module import (
    GeoTransformerEncoder,
    GeoDoubleAngleEncoder,
    MLPDecoder
)


# class LitGeoPretrainModel(pl.LightningModule):

#     def __init__(self, n_feats: int, n_cates: List[int], n_space_outs: int, n_time_outs: int, d_model: int,
#                  n_tf_head: int, n_tf_layer: int, p_tf_drop: float, n_mlp_layer: int, p_mlp_drop: float, lr: float):
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr

#         self.geo_encoder = GeoTransformerEncoder(n_feats, n_cates, d_model, n_tf_head, n_tf_layer, p_tf_drop)
#         self.space_decoder = MLPDecoder(d_model, n_space_outs, n_mlp_layer, p_mlp_drop)
        
#         if n_time_outs != 0:
#             self.time_decoder = MLPDecoder(d_model, n_time_outs, n_mlp_layer, p_mlp_drop)
#         else:
#             self.time_decoder = None

#         self.criterion = nn.MSELoss()

#     def forward(self, x_cont, x_cate):
#         x = self.geo_encoder(x_cont, x_cate)
#         if self.time_decoder:
#             return self.space_decoder(x), self.time_decoder(x)
#         return self.space_decoder(x), None

#     def training_step(self, batch, batch_idx):
#         space_outs, time_outs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
#         space_loss = self.criterion(space_outs, batch["SPACE_TARGET"])

#         if len(time_outs) > 0:
#             time_loss = self.criterion(time_outs, batch["TIME_TARGET"])
#             return space_loss + time_loss
#         else:
#             return space_loss

#     def validation_step(self, batch, batch_idx):
#         space_outs, time_outs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
#         space_loss = self.criterion(space_outs, batch["SPACE_TARGET"])
#         if len(time_outs) > 0:
#             time_loss = self.criterion(time_outs, batch["TIME_TARGET"])
#             total_loss = space_loss + time_loss
#         else:
#             total_loss = space_loss
#         self.log("valid_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr)

class LitGeoPretrainModel(pl.LightningModule):

    def __init__(self, n_feats: int, n_cates: List[int], n_space_outs: int, n_time_outs: int, d_model: int,
                 n_tf_head: int, n_tf_layer: int, p_tf_drop: float, n_mlp_layer: int, p_mlp_drop: float, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.geo_encoder = GeoTransformerEncoder(n_feats, n_cates, d_model, n_tf_head, n_tf_layer, p_tf_drop)
        self.space_decoder = MLPDecoder(d_model, n_space_outs, n_mlp_layer, p_mlp_drop)
        
        if n_time_outs != 0:
            self.time_decoder = MLPDecoder(d_model, n_time_outs, n_mlp_layer, p_mlp_drop)
        else:
            self.time_decoder = None

        self.criterion = nn.MSELoss()

    def forward(self, x_cont, x_cate):
        # 得到中间表示 x
        x = self.geo_encoder(x_cont, x_cate)
        # 利用 x 得到最终输出
        space_out = self.space_decoder(x)
        time_out = self.time_decoder(x) if self.time_decoder is not None else None
        # 返回 x 信息供下游任务使用，同时保持原有输出不变
        return space_out, time_out, x

    def training_step(self, batch, batch_idx):
        # 解包返回的三个变量，x 可在后续中使用，但损失计算依然基于前两个
        space_outs, time_outs, _ = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
        space_loss = self.criterion(space_outs, batch["SPACE_TARGET"])

        if time_outs is not None and len(time_outs) > 0:
            time_loss = self.criterion(time_outs, batch["TIME_TARGET"])
            return space_loss + time_loss
        else:
            return space_loss

    def validation_step(self, batch, batch_idx):
        space_outs, time_outs, _ = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
        space_loss = self.criterion(space_outs, batch["SPACE_TARGET"])
        if time_outs is not None and len(time_outs) > 0:
            time_loss = self.criterion(time_outs, batch["TIME_TARGET"])
            total_loss = space_loss + time_loss
        else:
            total_loss = space_loss
        self.log("valid_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LitGeoTaskModel(pl.LightningModule):

    def __init__(self, pretrain_ckpt_path: str, n_task_out: int, n_mlp_layer: int, p_mlp_drop: float, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.pretrain_model = LitGeoPretrainModel.load_from_checkpoint(pretrain_ckpt_path)
        self.task_decoder = MLPDecoder(self.pretrain_model.hparams["d_model"], n_task_out, n_mlp_layer, p_mlp_drop)

        self.criterion = nn.MSELoss()
        self.metric_r2 = R2Score(num_outputs=n_task_out)

    def forward(self, x_cont, x_cate):
        # 训练时只需要任务输出
        x = self.pretrain_model.geo_encoder(x_cont, x_cate)
        return self.task_decoder(x)

    def forward_with_features(self, x_cont, x_cate):
        # 新增方法：返回任务输出和中间特征 x
        x = self.pretrain_model.geo_encoder(x_cont, x_cate)
        task_out = self.task_decoder(x)
        return task_out, x

    def training_step(self, batch, batch_idx):
        outputs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
        loss = self.criterion(outputs, batch["TASK_TARGET"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
        loss = self.criterion(outputs, batch["TASK_TARGET"])

        self.metric_r2(outputs, batch["TASK_TARGET"])
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_r2", self.metric_r2, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        # 在预测时调用 forward_with_features 返回中间特征
        return self.forward_with_features(batch["CONT_FEAT"], batch["CATE_FEAT"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



# class LitGeoTaskModel(pl.LightningModule):

#     def __init__(self, pretrain_ckpt_path: str, n_task_out: int, n_mlp_layer: int, p_mlp_drop: float, lr: float):
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr

#         self.pretrain_model = LitGeoPretrainModel.load_from_checkpoint(pretrain_ckpt_path)
#         self.task_decoder = MLPDecoder(self.pretrain_model.hparams["d_model"], n_task_out, n_mlp_layer, p_mlp_drop)

#         self.criterion = nn.MSELoss()
#         self.metric_r2 = R2Score(num_outputs=n_task_out)

#     def forward(self, x_cont, x_cate):
#         x = self.pretrain_model.geo_encoder(x_cont, x_cate)
#         return self.task_decoder(x)

#     def training_step(self, batch, batch_idx):
#         outputs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
#         loss = self.criterion(outputs, batch["TASK_TARGET"])
#         return loss

#     def validation_step(self, batch, batch_idx):
#         outputs = self(batch["CONT_FEAT"], batch["CATE_FEAT"])
#         loss = self.criterion(outputs, batch["TASK_TARGET"])

#         self.metric_r2(outputs, batch["TASK_TARGET"])
#         self.log("valid_loss", loss, prog_bar=True)
#         self.log("valid_r2", self.metric_r2, on_step=False, on_epoch=True, prog_bar=True)

#     def predict_step(self, batch, batch_idx):
#         return self(batch["CONT_FEAT"], batch["CATE_FEAT"])

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr)


class LitGeoDoubleAngleModel(pl.LightningModule):

    def __init__(self, n_angle_feats: int, n_angle_data: int, n_normal_feats: int, n_normal_cates: List[int],
                 n_task_out: int, d_model: int, n_tf_head: int, n_tf_layer: int, p_tf_drop: float,
                 n_mlp_layer: int, p_mlp_drop: float, lr: float, double_angle: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.geo_encoder = GeoDoubleAngleEncoder(n_angle_feats, n_angle_data, n_normal_feats, n_normal_cates,
                                                 d_model, n_tf_head, n_tf_layer, p_tf_drop, double_angle)
        self.task_decoder = MLPDecoder(d_model, n_task_out, n_mlp_layer, p_mlp_drop)

        self.lr = lr
        self.criterion = nn.MSELoss()
        self.metric_r2 = R2Score(num_outputs=n_task_out)

    def forward(self, x_angle_feats_1, x_angle_feats_2, x_angle_data_1, x_angle_data_2, x_cont, x_cate):
        x = self.geo_encoder(x_angle_feats_1, x_angle_feats_2, x_angle_data_1, x_angle_data_2, x_cont, x_cate)
        return self.task_decoder(x)

    def _shared_step(self, batch):
        return self(batch["ANGLE_FEAT_1"], batch["ANGLE_FEAT_2"],
                    batch["ANGLE_DATA_1"], batch["ANGLE_DATA_2"],
                    batch["CONT_FEAT"], batch["CATE_FEAT"])

    def training_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        loss = self.criterion(outputs, batch["TASK_TARGET"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        loss = self.criterion(outputs, batch["TASK_TARGET"])

        self.metric_r2(outputs, batch["TASK_TARGET"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_r2", self.metric_r2, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self._shared_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        return optimizer
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=2,
        #     T_mult=2,
        #     eta_min=1e-5
        # )
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
