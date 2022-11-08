from argparse import Namespace
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from image_to_latex.data import Im2Latex
from image_to_latex.lit_models import LitResNetTransformer

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()

    lit_model = LitResNetTransformer(**cfg.lit_model)
    if(cfg.load_path!="None"):
        lit_model.load_from_checkpoint(cfg.load_path)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))
    loaderType=WandbLogger
    logger: Optional[loaderType] = None
    if cfg.logger:
#        logger = WandbLogger(**cfg.logger)
        logger = loaderType(**cfg.logger)


    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
#    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
