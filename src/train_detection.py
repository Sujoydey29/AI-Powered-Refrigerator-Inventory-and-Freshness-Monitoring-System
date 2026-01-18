import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11 on Fruits360")
    p.add_argument('--data',   type=str, default="data/fruits360.yaml", help="Path to dataset YAML")
    p.add_argument('--model',  type=str, default="yolo11n.pt",         help="Pretrained YOLO11 model")
    p.add_argument('--epochs', type=int, default=50,                   help="Number of epochs")
    p.add_argument('--imgsz',  type=int, default=256,                  help="Input image size (pixels)")
    p.add_argument('--batch',  type=int, default=16,                   help="Batch size")
    p.add_argument('--project',type=str, default="runs/train",         help="Save runs/project name")
    p.add_argument('--name',   type=str, default="fruits360",          help="Name of this run")
    return p.parse_args()

def main():
    args = parse_args()
    # Initialize a YOLO11 model (nano by default)
    model = YOLO(args.model)
    # Train!
    model.train(
    data    = args.data,
    epochs  = args.epochs,
    imgsz   = args.imgsz,
    batch   = args.batch,
    device  = '0',           # or 'cuda:0'
    project = args.project,
    name    = args.name
    )


if __name__ == "__main__":
    main()
