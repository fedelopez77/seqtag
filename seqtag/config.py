from pathlib import Path
import torch

PAD_LABEL = "<P>"
SUBTOKEN_LABEL = "<U>"
PAD_LABEL_ID = 0
SUBTOKEN_LABEL_ID = 1

TENSORBOARD_PATH = Path("tensorboard")
CKPT_PATH = Path("ckpt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BACKEND = "gloo"
if torch.cuda.is_available():
    torch.cuda.set_device(device=DEVICE)
    BACKEND = "nccl"

DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)
