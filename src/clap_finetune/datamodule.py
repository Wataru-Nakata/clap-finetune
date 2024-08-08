from pathlib import Path

import transformers
import webdataset as wds
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class AudioCaptionDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, model_name):
        super().__init__()
        train_urls = map(str, Path(train_path).glob("*.tar*"))
        val_urls = map(str, Path(val_path).glob("*.tar*"))
        self.train_dataset = (
            wds.WebDataset(train_urls, repeat=True)
            .decode(wds.autodecode.basichandlers, wds.torch_audio)
            .shuffle(100)
            .batched(8, collation_fn=self.collate_fn)
        )
        self.val_dataset = (
            wds.WebDataset(val_urls, repeat=True)
            .decode(wds.autodecode.basichandlers, wds.torch_audio)
            .batched(8, collation_fn=self.collate_fn)
        )
        self.model_name = model_name
        self.processor: transformers.ClapProcessor = (
            transformers.ClapProcessor.from_pretrained(self.model_name)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=None, num_workers=8)

    def collate_fn(self, batch):
        texts = []
        audios = []
        for sample in batch:
            texts.append(sample["json"]["text"])
            audio, sr = sample["flac"]
            audios.append(audio.view(-1).numpy())
        return self.processor(
            text=texts,
            audios=audios,
            return_tensors="pt",
            sampling_rate=sr,
            padding=True,
        )  # assume all audio files are sampled at the same sample rate
