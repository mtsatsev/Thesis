import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import pytorch_lightning as pl
from matplotlib.colors import to_rgb
from scipy import signal as ss
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def train_flow(flow, model_name="MNISTFlow"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(".",model_name),
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=200,
                         gradient_clip_val=1.0,
                         callbacks=[LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    train_data_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    print("Start training", model_name)
    trainer.fit(flow, train_data_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, test_dataloaders=val_loader, verbose=not USE_NOTEBOOK)
        start_time = time.time()
        test_result = trainer.test(flow, test_dataloaders=test_loader, verbose=not USE_NOTEBOOK)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}

    return flow, result
