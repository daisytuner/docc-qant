import torch
import cv2
import torchvision.models as models

import numpy as np

from pathlib import Path

import docc.torch
import tqdm

from docc.qant import register_docc_plugin

register_docc_plugin()

# BGR 
TARGET_COLORS = {
    "cuda": (0, 255, 0),  # Green
    "rocm": (0, 0, 255),     # Red
    "openmp": (255, 0, 0),   # Blue
    "qant": (0, 255, 255),     # Yellow
}
TARGET = "qant"

VIDEO_PATH = Path(__file__).parent / "video.MOV"
FRAMES_DIR = Path(__file__).parent / f"frames_{TARGET}"
FRAMES_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5

docc.torch.set_backend_options(target=TARGET, category="server")

if __name__ == "__main__":
    # Load the pre-trained Faster R-CNN model
    faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    faster_rcnn.eval()

    compiled_backbone = torch.compile(faster_rcnn.backbone, backend="docc")
    faster_rcnn.backbone = compiled_backbone

    # Open video
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer for final video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(str(FRAMES_DIR / "output.mp4"), fourcc, fps, (width, height))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dir = FRAMES_DIR / f"{i:04d}"
        frame_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(frame_dir / "frame.jpg"), frame)

        # BGR to RGB and normalization
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        if i % 3 == 0:
            # Perform object detection
            with torch.no_grad():
                detections = faster_rcnn(img_tensor)[0]
                last_detections = detections

            with open(frame_dir / "detections.txt", "w") as f:
                for box, score in zip(detections['boxes'], detections['scores']):
                    if score > CONFIDENCE_THRESHOLD:  # Filter out low-confidence detections
                        x1, y1, x2, y2 = box.int().tolist()
                        f.write(f"{x1} {y1} {x2} {y2} {score.item():.4f}\n")

        # Draw bounding boxes on the frame
        for box, score in zip(last_detections['boxes'], last_detections['scores']):
            if score > CONFIDENCE_THRESHOLD:  # Filter out low-confidence detections
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), TARGET_COLORS[TARGET], 4)

        out.write(frame)

        # Display the frame with detections
        # cv2.imshow('Detections', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        i = i + 1

cap.release()
out.release()
# cv2.destroyAllWindows()
