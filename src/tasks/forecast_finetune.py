import os
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from moment.common import PATHS
from moment.utils.utils import make_dir_if_not_exists, MetricsStore, dtype_map, EarlyStopping
from moment.utils.forecasting_metrics import get_forecasting_metrics
from .base import Tasks

from moment.models.moment import MOMENT
from moment.models.long_context_models.moment.infini_moment import InfiniMOMENT

warnings.filterwarnings('ignore')

class ForecastFinetuning(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        if 'truncate_by_padding' in self.args:
            self.global_mask_on_input = np.array([0] * self.args.seq_len, dtype=np.int64)
            self.global_mask_on_input[-self.args.truncation_by_padding_length:] = 1
        else:
            self.global_mask_on_input = np.array([1] * self.args.seq_len, dtype=np.int64)
        self.criterion = self._select_criterion(
            loss_type=self.args.loss_type, reduction='mean')
        
    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        
        truncation_padding_mask = torch.tensor(self.global_mask_on_input, device=self.device)
        
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader), desc='Metric Calculation'):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device) & truncation_padding_mask
                # todo: add here too
                forecast = batch_x.forecast.float().to(self.device)
    
                with torch.autocast(device_type='cuda', 
                                    dtype=dtype_map(self.args.torch_dtype), 
                                    enabled=self.args.use_amp):
                    outputs = self.model(x_enc=timeseries, 
                                         input_mask=input_mask, 
                                         mask=None)

                loss = self.criterion(outputs.forecast, forecast)                
                losses.append(loss.item())

                if return_preds:
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(outputs.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())
        
        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)
            return average_loss, losses, (trues, preds, histories)
        else:
            return average_loss
        
    def train(self):
        # Setup logger
        self.logger = self.setup_logger()
        self.run_name = self.logger.name
        self.dataset_name_ = self.args.dataset_names.split('/')[-1].split('.')[0]

        # Make necessary directories for logging and saving
        self.checkpoint_path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(self.checkpoint_path, verbose=True)
        self.optimizer = self._select_optimizer()

        # self.early_stopping = EarlyStopping(
        #     patience=self.args.patience, delta=self.args.delta)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)

        self.results_dir = self._create_results_dir(
            experiment_name="supervised_forecasting")
        
        # Load pre-trained MOMENT model before fine-tuning
        if not getattr(self.args, 'randomly_initialize_backbone', False):
            if self.args.model_name == 'MOMENT':
                self.load_pretrained_moment()

            elif self.args.model_name == 'InfiniMOMENT' and self.args.finetuning_mode =='infini-finetune-from-default':
                self.load_pretrained_moment(moment_class=MOMENT)
            elif self.args.model_name == 'InfiniMOMENT':
                self.load_pretrained_moment(moment_class=InfiniMOMENT)
            elif self.args.model_name.endswith('MOMENT'):
                if not getattr(self.args,'pretrained_multi_head', False):
                    if 'multi_chanel_encoder' in self.args:
                        self.load_pretrained_multi_moment_univariate()
                    else:
                        raise ValueError("Loading of pre-trained MultiMOMENT model (w/o multivariate channels pretrained) is not supported w/o the multi channel encoder config.")
                else:
                    self.load_pretrained_multi_moment_multivariate()
                    raise NotImplementedError("Loading of pre-trained MultiMOMENT model (w/ multivariate channels pretrained) is not implemented yet.")
        print("====== Frozen parameter status ======")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")
        self.model.to(self.device)

        # Evaluate the models before training
        eval_metrics = self.evaluate_and_log()
        
        opt_steps = 0
        cur_epoch = 0
        best_validation_loss = np.inf 
        
        truncation_padding_mask = torch.tensor(self.global_mask_on_input, device=self.device)
        while cur_epoch < self.args.max_epoch: # Epoch based learning only
            self.model.train()
            
            for batch_x in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f'Training Epoch {str(cur_epoch)} / {str(self.args.max_epoch)}'):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device) & truncation_padding_mask
                forecast = batch_x.forecast.float().to(self.device)
                
                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)
                
                with torch.autocast(device_type='cuda', 
                                    dtype=dtype_map(self.args.torch_dtype), 
                                    enabled=self.args.use_amp):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None)

                loss = self.criterion(outputs.forecast, forecast)
                
                if self.args.debug:
                    print(f"Step {opt_steps} loss: {loss.item()}")
                
                self.logger.log({"step_train_loss": loss.item(), 
                                 "learning_rate": self.optimizer.param_groups[0]['lr']})   

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.args.max_norm)
                
                if self.args.debug:
                    print(f"Gradient for model.layer1.weight: {self.model.layer1.weight.grad}")

                self.scaler.step(self.optimizer)
                
                # Updates the scale for next iteration.
                self.scaler.update()

                opt_steps = opt_steps + 1

                # Adjust learning rate
                if self.args.lr_scheduler_type == 'linearwarmupcosinelr':
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                elif self.args.lr_scheduler_type == 'onecyclelr': # Should be torch schedulers in general
                    self.lr_scheduler.step()
                    
            cur_epoch = cur_epoch + 1

            eval_metrics = self.evaluate_and_log()

            if eval_metrics.val_loss < best_validation_loss:
                best_validation_loss = eval_metrics.val_loss
                self.save_model_and_alert(opt_steps=None)
                
                forecasting_metrics = self.compute_forecasting_metrics(opt_steps=opt_steps)
                metrics_table = wandb.Table(
                    columns=forecasting_metrics.T.columns.tolist(),
                    data=forecasting_metrics.T.values.tolist()
                )
                self.logger.log({'forecasting_metrics': metrics_table})
                self.save_results(forecasting_metrics, self.results_dir, opt_steps)
            
        return self.model

    def compute_forecasting_metrics(self, opt_steps: int):
        _, _, (trues, preds, _) = self.validation(
            self.test_dataloader, return_preds=True)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    
        return pd.DataFrame(
            data = [self.run_name, self.logger.id, opt_steps, 
                    metrics.mae, metrics.mse, metrics.mape, 
                    metrics.smape, metrics.rmse],
            index = ['Model name', 'ID', 'Opt. steps', 
                     'MAE', 'MSE', 'MAPE', 'sMAPE', 'RMSE'],
        )
    
        return eval_metrics


