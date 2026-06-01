import os
from config import CHECKPOINT_DIR
from train import TrainerB

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    TrainerB().run()
