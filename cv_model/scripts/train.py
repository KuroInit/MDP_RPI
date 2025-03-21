import argparse
import wandb
from ultralytics import YOLO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.train_config import train_config
from ultralytics import settings

settings.update({"wandb": True})

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Train Script")
  
    parser.add_argument("--model", type=str, default=None, help="Initial weights path, e.g. yolov8n.pt")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Resume training from a given checkpoint (e.g. 'runs/detect/exp/weights/last.pt')")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model is not None:
        train_config["model"] = args.model

    # Resume training
    if args.resume is not None:
        train_config["model"] = args.resume
        resume_training = True
    else:
        resume_training = False

    model = YOLO(train_config["model"])
    print(f"Model: {model}")

    results = model.train(
        data=train_config["data"],
        epochs=train_config["epochs"],
        batch=train_config["batch_size"],
        imgsz=train_config["img_size"],
        project=train_config["project"],
        name=train_config["name"],
        optimizer=train_config["optimizer"],
        lr0=train_config["lr0"],
        device=train_config["device"],
        workers=train_config["workers"],
        seed=train_config["seed"],
        resume=resume_training,   
        verbose=train_config["verbose"],
        save_period=train_config["save_period"],           
    )

    # Val and Test the model
    model.val(
        data=train_config["data"],
        batch=train_config["batch_size"],
        imgsz=train_config["img_size"]
    )


if __name__ == "__main__":
    main()
