from pytorch_lightning import Trainer

import clap_finetune
import clap_finetune.datamodule
import clap_finetune.lightning_module

if __name__ == "__main__":
    lightning_module = clap_finetune.lightning_module.CLAPLightningModule()
    data_module = clap_finetune.datamodule.AudioCaptionDataModule(
        "/home/wnakata/cropped-wavcaps/notebooks/dcase_bbc1_no/train",
        "/home/wnakata/cropped-wavcaps/notebooks/dcase_bbc1_no/test",
        lightning_module.model_name,
    )
    trainer = Trainer(profiler="pytorch")
    trainer.fit(lightning_module, datamodule=data_module)
