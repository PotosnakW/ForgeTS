import os
import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import wandb
from wandb import AlertLevel

from moment.common import PATHS
from moment.utils.utils import MetricsStore
from moment.utils.optims import LinearWarmupCosineLRScheduler
from moment.utils.forecasting_metrics import sMAPELoss
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.base import BaseModel
#from moment.models.moment import MOMENT
from moment.models.moment import MOMENT

from moment.models.long_context_models.UP2ME.UP2ME import UP2MEforecasterMOMENT
from moment.models.long_context_models.TimeSeriesLibrary.itransformer import iTransformer
from moment.models.long_context_models.TimeSeriesLibrary.crossformer import CrossFormer

from moment.utils.onecycleLR_different_lr import OneCycleLRDifferentLR
warnings.filterwarnings('ignore')

class Tasks(nn.Module):
    def __init__(self, args, **kwargs):
        super(Tasks, self).__init__()
        self.args = args
        self._dataloader = {}
        
        self._acquire_device()
        
        # Setup data loaders
        self.train_dataloader = self._get_dataloader(data_split='train')
        self.val_dataloader = self._get_dataloader(data_split='val')
        
        if self.args.task_name != "pre-training":
            self.test_dataloader = self._get_dataloader(data_split='test')
        
        self._set_dataloader_metrics_to_args()
        
        self._build_model()

    def _set_dataloader_metrics_to_args(self):
        return None
        
        
    def _build_model(self):
        if self.args.model_name == 'MOMENT':
            self.model = MOMENT(configs=self.args)
        elif self.args.model_name == 'UP2MEforecasterMOMENT':
            self.model = UP2MEforecasterMOMENT(configs=self.args)
        elif self.args.model_name == 'iTransformer':
            self.model = iTransformer(configs=self.args)
        elif self.args.model_name == 'CrossFormer':
            self.model = CrossFormer(configs=self.args)
        else:
            raise NotImplementedError(f"Model {self.args.model_name} not implemented")
        return self.model
    
    def _acquire_device(self):
        self.device = torch.device('cuda:{}'.format(self.args.device))
        return self.device

    def _reset_dataloader(self):
        self._dataloader = {}
        
    def log_beta_parameters(self, betas, i):
        beta_values = betas.detach().cpu().numpy()  # Convert to numpy array for logging
        for j, value in enumerate(beta_values.flatten()):
            self.logger.log({f"beta_{i}_{j}": value})

    def _get_dataloader(self, data_split : str = 'train'):
        # Load Datasets
        if self._dataloader.get(data_split) is not None:
            return self._dataloader.get(data_split)
        else:
            data_loader_args = deepcopy(self.args)
            data_loader_args.data_split = data_split
            if self.args.task_name == "pre-training":
                data_loader_args.dataset_names = "all"
            data_loader_args.batch_size =\
                self.args.train_batch_size if data_split == 'train' else self.args.val_batch_size
            print(f"Loading {data_split} split of the dataset")
            
            self._dataloader[data_split] = get_timeseries_dataloader(args=data_loader_args)
            return self._dataloader.get(data_split)
        
    def _select_optimizer(self):
        # Extract beta params
        beta_params = [param for key, param in self.model.named_parameters() if 'beta' in key]
        # Extract the rest of the parameters
        non_beta_params = [param for key, param in self.model.named_parameters() if 'beta' not in key]
        
        self.args.beta_lr = self.args.beta_lr if hasattr(self.args, 'beta_lr') else self.args.init_lr

        # Define the optimizer with parameter groups
        if self.args.optimizer_name == "AdamW":
            optimizer = optim.AdamW([
                {'params': non_beta_params, 'lr': self.args.init_lr, 'weight_decay':self.args.weight_decay},  # Default learning rate
                {'params': beta_params, 'lr': self.args.beta_lr, 'weight_decay': 0}       # Specific learning rate for beta params
            ])
        elif self.args.optimizer_name == "Adam":
            optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.args.init_lr,
                                   weight_decay=self.args.weight_decay)
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), 
                                  lr=self.args.init_lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer_name} not implemented")
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            print(f"Parameter Group {i} with lr = {lr}:")
            for param in param_group['params']:
                print(f"  - Parameter ID: {id(param)}")
        return optimizer
    
    def _init_lr_scheduler(self, type : str = 'linearwarmupcosinelr'):
        decay_rate = self.args.lr_decay_rate
        warmup_start_lr = self.args.warmup_lr
        warmup_steps = self.args.warmup_steps

        if type == 'linearwarmupcosinelr':
            self.lr_scheduler = LinearWarmupCosineLRScheduler(
                optimizer=self.optimizer,
                max_epoch=self.args.max_epoch,
                min_lr=self.args.min_lr,
                init_lr=self.args.init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        elif type == 'onecyclelr':
            self.lr_scheduler = OneCycleLRDifferentLR(
                optimizer=self.optimizer,
                max_lr=self.args.init_lr,
                epochs=self.args.max_epoch,
                steps_per_epoch=len(self.train_dataloader),
                pct_start=self.args.pct_start,
            )
        elif type == 'none':
            self.lr_scheduler = None

    def _select_criterion(self, 
                          loss_type : str = 'mse',
                          reduction : str = 'none',
                          **kwargs):
        if loss_type == 'mse':
            criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'mae':
            criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'huber':
            criterion = nn.HuberLoss(reduction=reduction,
                                     delta=kwargs['delta'])
        elif loss_type =="smape":
            criterion = sMAPELoss(reduction=reduction)
        if loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(reduction=reduction)
        return criterion
    
    def save_results(self, 
                     results_df: pd.DataFrame,
                     path: str,
                     opt_steps: int):
        results_df.to_csv(os.path.join(path, f'results_{self.args.task_name}_{opt_steps}.csv'))
    
    def save_model(self, 
                   model: nn.Module, 
                   path: str,
                   opt_steps: int, 
                   optimizer: torch.optim.Optimizer, 
                   scaler: torch.cuda.amp.GradScaler):
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
            }

        if opt_steps is None:
            with open(os.path.join(path, f'{self.args.model_name}.pth'), 'wb') as f:
                torch.save(checkpoint, f)
        else:
            with open(os.path.join(path, f'{self.args.model_name}_checkpoint_{opt_steps}.pth'), 'wb') as f:
                torch.save(checkpoint, f)
    
    def save_model_and_alert(self, opt_steps):
        self.save_model(
            self.model, self.checkpoint_path, 
            opt_steps, self.optimizer, self.scaler)
        
    def load_pretrained_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, 
                                map_location = lambda storage, loc: storage.cuda(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # TODO: Load the latest checkpoint

    def load_pretrained_moment(self, 
                             pretraining_task_name: str = "pre-training",
                             do_not_copy_head: bool = True,
                             moment_class = MOMENT):
        def get_class_name(full_class_str):
            return full_class_str.split('.')[-1].rstrip("'>")

        pretraining_args = deepcopy(self.args)
        pretraining_args.task_name = pretraining_task_name
            
        print(f"Checkpoint: {pretraining_args.pretraining_run_name}")
        checkpoint = BaseModel.load_pretrained_weights(
            run_name=pretraining_args.pretraining_run_name, 
            opt_steps=pretraining_args.pretraining_opt_steps,
            model_name=get_class_name(str(moment_class)))
        
        pretrained_model = moment_class(configs=pretraining_args)
        pretrained_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        # Create a dictionary for quick lookup of pretrained parameters
        pretrained_params = {name: param for name, param in pretrained_model.named_parameters()}

        # Iterate over the model's parameters and match them to the pretrained parameters
        for name_f, param_f in self.model.named_parameters():
            if name_f in pretrained_params:
                param_p = pretrained_params[name_f]
                if param_p.shape == param_f.shape:
                    if do_not_copy_head and name_f.startswith("head"):
                        continue
                    else:
                        param_f.data = param_p.data
                
        self.freeze_model_parameters() # Freeze model parameters based on fine-tuning mode
        
        return True
        
    def freeze_model_parameters(self):
        if self.args.finetuning_mode == 'linear-probing':
            for name, param in self.model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        elif self.args.finetuning_mode == 'infini-finetune-from-default':
            # here we need to fine-tune only the channel embeddings, the 
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if (name.startswith("head")) or (name.startswith("static_channel_embedding")) or (name.endswith("beta")):
                    param.requires_grad = True
        elif self.args.finetuning_mode == 'linear-probing-static':
            # here we need to fine-tune only the channel embeddings, the 
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if (name.startswith("head")) or (name.startswith("static_channel_embedding")):
                    param.requires_grad = True
        elif self.args.finetuning_mode == 'linear-probing-infini-beta':
            # here we need to fine-tune only the channel embeddings, the 
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if (name.startswith("head")) or (name.startswith("static_channel_embedding")) or (name.endswith("beta")):
                    param.requires_grad = True             
        elif self.args.finetuning_mode == 'end-to-end':
            pass
        elif self.args.finetuning_mode == 'multivariate': #TODO: check if nicely initiated!
            for name, param in self.model.named_parameters():
                if name.startswith("head") or name.startswith('channel_encoding') or name.startswith('sequence_encoding') :
                    param.requires_grad = True
                elif name.startswith('temporal_channel_layers') or name.startswith('enc_2_dec') or name.startswith('position_embedding') or name.startswith('channel_embedding'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif self.args.finetuning_mode == 'multivariate-zero-shot-head': #TODO: check if nicely initiated!
            for name, param in self.model.named_parameters():
                if name.startswith('channel_encoding') or name.startswith('sequence_encoding') :
                    param.requires_grad = True
                elif name.startswith('temporal_channel_layers') or name.startswith('enc_2_dec') or name.startswith('position_embedding') or name.startswith('channel_embedding'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        else:
            raise NotImplementedError(
                f"Finetuning mode {self.args.finetuning_mode} not implemented")
        
        print("====== Frozen parameter status ======")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Not frozen:", name)
            else:
                print("Frozen:", name)
        print("=====================================")

    def _create_results_dir(self, experiment_name="supervised_forecasting"):
        
        if experiment_name == "supervised_forecasting":
            results_path = os.path.join(
                PATHS.RESULTS_DIR, experiment_name, 
                self.args.model_name, self.dataset_name_, 
                self.args.finetuning_mode, 
                str(self.args.forecast_horizon))
        elif experiment_name == "supervised_anomaly_detection":
            results_path = os.path.join(
                PATHS.RESULTS_DIR, experiment_name, 
                self.args.model_name,
                self.args.finetuning_mode)
        elif experiment_name == "supervised_imputation":
            results_path = os.path.join(
                PATHS.RESULTS_DIR, experiment_name, 
                self.args.model_name, self.dataset_name_, 
                self.args.finetuning_mode)
        elif experiment_name == "supervised_classification":
            results_path = os.path.join(
                PATHS.RESULTS_DIR, experiment_name, 
                self.args.model_name, self.dataset_name_, 
                self.args.finetuning_mode)

        os.makedirs(results_path, exist_ok=True)
        return results_path
                
    def setup_logger(self, notes: str = None):
        self.logger = wandb.init(
            project="Project Name",
            dir=PATHS.WANDB_DIR,
            config=self.args,
            name=self.args.run_name if hasattr(self.args, 'run_name') else None,
            notes=self.args.notes if notes is None else notes,
            mode='disabled' if self.args.debug else 'run',
            entity="ninazukowska"
        )
        artifact = wandb.Artifact(name="experiment_config", type="code")
        artifact.add_file(local_path=self.args.experiment_config_path)
        self.logger.log_artifact(artifact)  
        
        if self.args.debug:
            print(f'Run name: {self.logger.name}\n')
        return self.logger
    
    def end_logger(self):
        self.logger.finish()
    
    def evaluate_model(self):
        return MetricsStore(
            train_loss = self.validation(self.train_dataloader),
            test_loss = self.validation(self.test_dataloader),
            val_loss = self.validation(self.val_dataloader)
        )
    
    def evaluate_and_log(self):
        eval_metrics = self.evaluate_model()
        self.logger.log({
            "train_loss": eval_metrics.train_loss,
            "validation_loss": eval_metrics.val_loss,
            "test_loss": eval_metrics.test_loss
            })
        # get named params and if any betas are present - log them
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if 'beta' in name:
                self.log_beta_parameters(param, i)
        return eval_metrics
    
    def debug_model_outputs(self, loss, outputs, batch_x, **kwargs):
        # Debugging code
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)) or (loss < 1e-3):
            self.logger.alert(title="Loss is NaN or Inf or too small",
                                text=f"Loss is {loss.item()}.", 
                                level=AlertLevel.INFO)
            breakpoint() 

        # Check model outputs
        if outputs.illegal_output:
            self.logger.alert(title="Model weights are NaN or Inf",
                                text=f"Model weights are NaN or Inf.", 
                                level=AlertLevel.INFO)
            breakpoint()

        # Check model gradients
        illegal_encoder_grads = torch.stack(
            [torch.isfinite(p).any() for p in self.model.encoder.parameters()]).any().item()
        illegal_head_grads = torch.stack(
            [torch.isfinite(p).any() for p in self.model.head.parameters()]).any().item()
        illegal_patch_embedding_grads = torch.stack(
            [torch.isfinite(p).any() for p in self.model.patch_embedding.parameters()]).any().item()
        
        illegal_grads = illegal_encoder_grads or illegal_head_grads or illegal_patch_embedding_grads

        if illegal_grads:
            self.logger.alert(title="Model gradients are NaN or Inf",
                                text=f"Model gradients are NaN or Inf.", 
                                level=AlertLevel.INFO)
            # breakpoint()
        
        return
        
