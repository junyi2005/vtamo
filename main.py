import argparse
import datetime
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer import Trainer

from utils.helpers import instantiate_from_config
from vtamo.callbacks import SetupCallback


def str2bool(v: Any) -> bool:
    """Convert string representation to boolean.
    
    Args:
        v: Input value to convert
        
    Returns:
        Boolean representation of input
        
    Raises:
        ArgumentTypeError: If input cannot be interpreted as boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser with all required CLI options.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description='VTaMo training')
    parser.add_argument(
        '-c', '--config', nargs='*', metavar='base_config.yaml', default=list(),
        help='Configuration files to load'
    )
    parser.add_argument(
        '-t', '--train', type=str2bool, default=True, nargs='?',
        help='Run in training mode'
    )
    parser.add_argument(
        '--test', type=bool, default=False,
        help='Run in testing mode'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='Seed for random number generators'
    )
    parser.add_argument(
        '-f', '--fast_dev_run', action='store_true', default=False,
        help='Run a test batch for debugging'
    )
    parser.add_argument(
        '-n', '--name', type=str, const=True, default='', nargs='?',
        help='Postfix for log directory'
    )
    parser.add_argument(
        '--postfix', type=str, default='',
        help='Additional postfix for log directory'
    )
    parser.add_argument(
        '-l', '--logdir', type=str, default='logs',
        help='Base directory for logging'
    )
    parser.add_argument(
        '-r', '--resume', default=None,
        help='Resume training from checkpoint directory'
    )
    parser.add_argument(
        '--no_test', type=bool, default=True,
        help='Skip test phase after training'
    )
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='Checkpoint file for resuming or testing'
    )
    parser.add_argument(
        '-e', '--evaluation', type=str, default='mse',
        help='Evaluation metric to use'
    )
    parser.add_argument(
        '--attn_pool', type=str2bool, default=None, nargs='?', const=True,
        help='Use AttentionTemporalConv instead of TemporalConv (learnable downsampling)'
    )
    parser.add_argument(
        '--warmup', type=int, default=None,
        help='Force warm_up_steps to this value, overriding the config.'
    )
    parser.add_argument(
        '--auto_warmup', action='store_true',
        help='Auto-scale warm_up_steps from the actual training-set size instead of '
             'using the config value: warmup = clamp(warmup_ratio * max_epochs * '
             'N_train / effective_batch, [warmup_min, warmup_max]). Useful for small '
             'runs where the config value would never finish. Off by default: the '
             'config is authoritative, and warm_up_steps also gates the global '
             'alignment schedule.'
    )
    parser.add_argument(
        '--warmup_ratio', type=float, default=0.016,
        help='Fraction of total training steps used as warmup when --auto_warmup is '
             'set (default 0.016, matches 40000 / (500 ep * 5000 steps)).'
    )
    parser.add_argument(
        '--warmup_min', type=int, default=100,
        help='Lower bound for auto-scaled warmup (default 100 steps).'
    )
    parser.add_argument(
        '--warmup_max', type=int, default=40000,
        help='Upper bound for auto-scaled warmup (default 40000 matches the '
             'value used for the reported runs).'
    )
    return parser


def load_configs(config_paths: List[str]) -> OmegaConf:
    """Load and merge multiple configuration files.
    
    Args:
        config_paths: List of paths to configuration files
        
    Returns:
        Merged configuration
    """
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    return OmegaConf.merge(*configs)


def setup_logging_dirs(opt: argparse.Namespace) -> tuple:
    """Set up logging directories and determine checkpoint path.
    
    Args:
        opt: Command line arguments
        
    Returns:
        Tuple of (logdir, checkpoint_path, nowname)
        
    Raises:
        ValueError: If resuming from a non-existent directory
    """
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find checkpoint directory: {opt.resume}")
            
        logdir = opt.resume.rstrip("/")
        if opt.ckpt:
            ckpt = os.path.join(logdir, "checkpoints", opt.ckpt)
        else:
            # Auto-find last checkpoint for resuming
            last_ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt = last_ckpt
            else:
                # Fallback: find the most recent checkpoint
                ckpt_files = sorted(glob.glob(os.path.join(logdir, "checkpoints", "*.ckpt")))
                ckpt = ckpt_files[-1] if ckpt_files else None
                if ckpt is None:
                    print(f"WARNING: --resume specified but no checkpoints found in {logdir}/checkpoints/")
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        ckpt = opt.ckpt
    
    return logdir, ckpt, nowname


def configure_callbacks(
    opt: argparse.Namespace, 
    model: pl.LightningModule, 
    ckptdir: str, 
    lightning_config: OmegaConf,
    logdir: str,
    now: str,
    config: OmegaConf
) -> List:
    """Configure training callbacks.
    
    Args:
        opt: Command line arguments
        model: Lightning module
        ckptdir: Directory for checkpoints
        lightning_config: Lightning configuration
        logdir: Directory for logs
        now: Current timestamp
        config: Full configuration
        
    Returns:
        List of callbacks
    """
    callbacks = [
        instantiate_from_config(lightning_config.callback[callback]) 
        for callback in lightning_config.callback.keys()
    ]
    
    # Always save both best BLEU4 and best loss checkpoints
    # Checkpoint for best BLEU4 (higher is better)
    callbacks.append(ModelCheckpoint(
        dirpath=ckptdir,
        filename="best_bleu4-epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}",
        monitor="val/bleu4",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="max"
    ))

    # Checkpoint for best loss (lower is better)
    loss_monitor = "val/loss"
    callbacks.append(ModelCheckpoint(
        dirpath=ckptdir,
        filename="best_loss-epoch={epoch:05}-step={step:07}-loss={val/loss:.4f}",
        monitor=loss_monitor,
        auto_insert_metric_name=False,
        save_top_k=1,
        mode="min"
    ))

    # Checkpoint for latest (always save the most recent)
    callbacks.append(ModelCheckpoint(
        dirpath=ckptdir,
        filename="last",
        monitor=None,  # No metric, just save latest
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,  # We're manually naming it "last"
    ))

    # Early stopping based on evaluation metric
    if opt.evaluation == "bleu":
        callbacks.append(EarlyStopping(
            monitor="val/bleu4", verbose=True, patience=50, mode="max"
        ))
    else:
        callbacks.append(EarlyStopping(
            monitor=loss_monitor, verbose=True, patience=50, mode="min"
        ))
    
    # Setup callback for logging configuration
    callbacks.append(SetupCallback(
        resume=opt.resume, 
        now=now, 
        logdir=logdir, 
        ckptdir=ckptdir, 
        cfgdir=os.path.join(logdir, "configs"),
        config=config, 
        lightning_config=lightning_config
    ))
    
    return callbacks


def configure_logger(logger_type: str, logdir: str, nowname: str) -> Dict:
    """Configure the logger.
    
    Args:
        logger_type: Type of logger to use
        logdir: Directory for logs
        nowname: Name for the current run
        
    Returns:
        Logger configuration
    """
    logger_configs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "project": "vtamo",
                "save_dir": logdir,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "version": nowname,
                "save_dir": logdir
            }
        }
    }
    
    if logger_type not in logger_configs:
        logger_type = "tensorboard"
        
    return logger_configs[logger_type]


def main():
    """Main entry point for training and testing."""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    
    # Parse arguments
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    
    # Validate arguments
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    # Set up directories and checkpoint path
    logdir, ckpt, nowname = setup_logging_dirs(opt)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # Set random seed for reproducibility
    seed_everything(opt.seed)
    
    # Load configuration files
    if opt.resume or opt.test:
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config

    config = load_configs(opt.config)
    lightning_config = config.pop("lightning", OmegaConf.create())

    # Override use_attention_pool from command line if specified
    if opt.attn_pool is not None:
        if "params" in config.model:
            config.model.params.use_attention_pool = opt.attn_pool
            print(f"[CLI Override] use_attention_pool = {opt.attn_pool}")
    
    # Configure trainer
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    if opt.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    
    # Instantiate data module
    data = instantiate_from_config(config.data)
    data.setup()

    # ------------------------------------------------------------------
    # warm_up_steps resolution.
    #
    # By default the CONFIG IS AUTHORITATIVE — warm_up_steps is left exactly as the
    # yaml specifies. This matters because warm_up_steps is not only the LM-freeze
    # switch: it is also the Stage 0 -> Stage 1 boundary of the global alignment
    # schedule (and triggers the Procrustes init of T), so silently rescaling it
    # would change the method.
    #
    # --warmup <N>    forces a value.
    # --auto_warmup   opts in to scaling from the actual training-set size, for small
    #                 runs where a large configured warmup would never finish:
    #                   effective_batch = batch_size * accumulate_grad_batches
    #                   total_steps     = max(1, max_epochs * N_train // effective_batch)
    #                   warmup          = clamp(warmup_ratio * total_steps,
    #                                           [warmup_min, warmup_max])
    # ------------------------------------------------------------------
    if "params" in config.model:
        model_params = config.model.params
        train_ds = data.datasets.get("train") if hasattr(data, "datasets") else None
        n_train = len(train_ds) if train_ds is not None else 0

        trainer_cfg = lightning_config.get("trainer", OmegaConf.create())
        batch_size = int(config.data.params.get("batch_size", 1))
        accum = int(trainer_cfg.get("accumulate_grad_batches", 1))
        max_epochs = int(trainer_cfg.get("max_epochs", 1))
        effective_batch = max(1, batch_size * accum)

        if opt.warmup is not None:
            chosen = int(opt.warmup)
            source = "CLI --warmup"
        elif opt.auto_warmup and n_train > 0:
            total_steps = max(1, max_epochs * n_train // effective_batch)
            raw = int(round(total_steps * opt.warmup_ratio))
            chosen = max(opt.warmup_min, min(opt.warmup_max, raw))
            source = (f"auto (N_train={n_train}, batch={batch_size}*accum={accum}, "
                      f"max_epochs={max_epochs}, total_steps={total_steps}, "
                      f"ratio={opt.warmup_ratio})")
        else:
            chosen = int(model_params.get("warm_up_steps", opt.warmup_min))
            source = "config"

        original = model_params.get("warm_up_steps", None)
        model_params.warm_up_steps = chosen
        print(f"[warmup] warm_up_steps: {original} -> {chosen}  [{source}]",
              flush=True)

    # Instantiate model
    model = instantiate_from_config(config.model)
    
    # Configure trainer with callbacks and logger for non-dev runs
    if not opt.fast_dev_run:
        logger_cfg = configure_logger("wandb", logdir, nowname)
        trainer_opt.logger = instantiate_from_config(logger_cfg)
        
        trainer_opt.callbacks = configure_callbacks(
            opt, model, ckptdir, lightning_config, logdir, now, config
        )
    
    # Create trainer
    trainer = Trainer(**vars(trainer_opt))
    
    # Run training or testing
    if opt.train:
        if opt.resume is not None:
            trainer.fit(model, data, ckpt_path=ckpt)
        else:
            if ckpt is not None:
                model.load_pretrained_weights(ckpt)
                trainer.fit(model, data)
            else:
                trainer.fit(model, data)
            
            if not opt.no_test:
                trainer.test(model, data)
    elif opt.test:
        trainer.test(model, data, ckpt_path=ckpt)


if __name__ == '__main__':
    main()
