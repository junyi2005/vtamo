import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.helpers import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, num_workers=None):
        super().__init__()

        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['valid'] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        
    def _train_dataloader(self):
        return DataLoader(
            dataset=self.datasets['train'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # collate_fn=BaseFeeder.collate_fn if self.use_collate else None
            collate_fn=self.datasets['train'].collate_fn
        )

    def _val_dataloader(self):
        return DataLoader(
            dataset=self.datasets['valid'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=BaseFeeder.collate_fn if self.use_collate else None,
            collate_fn=self.datasets['valid'].collate_fn
        )

    def _test_dataloader(self):
        return DataLoader(
            dataset=self.datasets['test'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=BaseFeeder.collate_fn if self.use_collate else None
            collate_fn=self.datasets['test'].collate_fn
        ) 