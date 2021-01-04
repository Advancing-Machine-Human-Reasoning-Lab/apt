import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration


class T5Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def forward(self):
        pass


model = T5Model()
ckpt = torch.load("t5_paraphrase1/checkpointepoch=2.ckpt")
model.load_state_dict(ckpt["state_dict"])

# save the inner pretrained model
model.model.save_pretrained("t5_paraphrase1/model")
