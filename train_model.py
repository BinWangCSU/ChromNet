import sys
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

import ChromNet_model
from data import genome_dataset as genome_dataset

def main():
    args = init_parser()
    init_training(args)

def init_parser():
  parser = argparse.ArgumentParser(description='Training Module.')

  # Data and Run Directories
  parser.add_argument('--seed', dest='run_seed', default=2077,
                        type=int,
                        help='Random seed for training')
  parser.add_argument('--save_path', dest='run_save_path', default='checkpoints',
                        help='Path to the model checkpoint')

  parser.add_argument('--data-root', dest='dataset_data_root', default='data',
                        help='Root path of training data', required=True)
  parser.add_argument('--assembly', dest='dataset_assembly', default='hg38',
                        help='Genome assembly for training data')
  parser.add_argument('--celltype', dest='dataset_celltype', nargs='+', default=['IMR90' 'K562' 'IMR90_noise'],
                    help='Sample cell types for prediction, used for output separation')

  parser.add_argument('--model-type', dest='model_type', default='ChromNet')

  # Training Parameters
  parser.add_argument('--patience', dest='trainer_patience', default=80,
                        type=int,
                        help='Epoches before early stopping')
  parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=80,
                        type=int,
                        help='Max epochs')
  parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=20,
                        type=int,
                        help='Top n models to save')
  parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=4,
                        type=int,
                        help='Number of GPUs to use')

  parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, 
                        type=int,
                        help='Batch size')
  parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled',
                        action='store_false',
                        help='Using ddp, adjust batch size')
  parser.add_argument('--num-workers', dest='dataloader_num_workers', default=16,
                        type=int,
                        help='Dataloader workers')


  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  return args

def init_training(args):

    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer_patience,
                                        verbose=False,
                                        mode="min")

    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                        save_top_k=args.trainer_save_top_n, 
                                        monitor='val_loss')

    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run_save_path}/csv')
    all_loggers = csv_logger
    
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)

    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')

    pl_trainer = pl.Trainer(
                            accelerator="gpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger=all_loggers,
                            callbacks=[early_stop_callback,
                                       checkpoint_callback,
                                       lr_monitor],
                            max_epochs=args.trainer_max_epochs
                            )
    pl_trainer.fit(pl_module, trainloader, valloader)


class TrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model(args)
        self.args = args
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        seq, features, mat, start, end, cell_type, chr_name, chr_idx = batch
        features = torch.cat([feat.unsqueeze(2) for feat in features], dim = 2)
        inputs = torch.cat([seq, features], dim = 2)
        mat = mat.float()
        return inputs, mat, cell_type
    
    def training_step(self, batch, batch_idx):
        inputs, mat, cell_type = self.proc_batch(batch)
        outputs, classification_output = self(inputs)

        hic_mask = (cell_type == 0) | (cell_type == 2)
        classification_mask = (cell_type == 0) | (cell_type == 1)

        real_outputs = outputs[hic_mask]
        real_mat = mat[hic_mask]

        if real_outputs.shape[0] == 0:
            hic_loss = torch.tensor(0.0, requires_grad=True)
        else:
            hic_loss = torch.nn.MSELoss()(real_outputs, real_mat)

        class_output=classification_output[classification_mask]
        cell_label=cell_type[classification_mask]

        if class_output.shape[0] == 0:
            classification_loss = torch.tensor(0.0, requires_grad=True)
        else:
            classification_loss = torch.nn.CrossEntropyLoss()(class_output, cell_label)

        loss = hic_loss + classification_loss
        metrics = {'train_step_loss': loss, 'hic_loss': hic_loss, 'classification_loss': classification_loss}

        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat, cell_type = self.proc_batch(batch)
        
        outputs, _ = self(inputs)

        hic_mask = (cell_type == 0) | (cell_type == 2)
        real_outputs = outputs[hic_mask]
        real_mat = mat[hic_mask]

        if real_outputs.shape[0] == 0:
            hic_loss = torch.tensor(0.0, requires_grad=True)
        else:
            hic_loss = torch.nn.MSELoss()(real_outputs, real_mat)

        loss = hic_loss

        return loss

    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr = 2e-4,
                                     weight_decay = 0)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}

    def get_dataset(self, args, mode):

        all_celltype_roots=[]
        all_genomic_features=[]
        for dataset_celltype in args.dataset_celltype:
            celltype_root = f'{args.dataset_data_root}/{args.dataset_assembly}/{dataset_celltype}'
            genomic_features = {'ctcf' : {'file_name' : 'ctcf.bw', 'norm' : 'log' },
                                'atac' : {'file_name' : 'atac.bw', 'norm' : 'log' }}

            all_celltype_roots.append(celltype_root)
            all_genomic_features.append(genomic_features)

        dataset = genome_dataset.GenomeDataset(args.dataset_celltype,
                                all_celltype_roots, 
                                args.dataset_assembly,
                                all_genomic_features, 
                                mode = mode,
                                include_sequence = True,
                                include_genomic_features = True)

        if mode == 'val':
            self.val_length = len(dataset) / args.dataloader_batch_size
            print('Validation loader length:', self.val_length)

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        if mode == 'train':
            shuffle = True
        else: 
            shuffle = False
        
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers

        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus) 

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,

            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
        return dataloader


    def get_model(self, args):
        model_name =  args.model_type
        num_genomic_features = 7
        num_cell_types = 2
        ModelClass = getattr(ChromNet_model, model_name)
        model = ModelClass(num_genomic_features, num_cell_types)

        return model

if __name__ == '__main__':
    main()
