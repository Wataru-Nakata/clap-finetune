import torch
import transformers
from pytorch_lightning import LightningModule


class CLAPLightningModule(LightningModule):
    def __init__(self, model_name="laion/clap-htsat-fused"):
        super().__init__()
        self.model: transformers.ClapModel = transformers.ClapModel.from_pretrained(
            model_name
        )
        self.model_name = model_name
        self.save_hyperparameters()

    def forward(self, inputs):
        output = self.model(**inputs, return_loss=True)
        return output

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        output = self(batch)
        self.log("train/loss", output.loss)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        output = self(batch)
        self.log("val/loss", output.loss)
        return output.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
