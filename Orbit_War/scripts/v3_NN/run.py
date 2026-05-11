import os
from config import CHECKPOINT_DIR
from train import Trainer

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    trainer = Trainer()
    trainer.run()