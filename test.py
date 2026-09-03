import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESH = 0.5   # detection confidence threshold
MASK_THRESH       = 0.5   # mask binarization threshold
FRAME_DOWNSCALE   = 1.0   # set <1.0 to speed up processing
# ───────────────────────────────────────────────────────────────────────────────

def load_model(device):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)
    return model

def preprocess_image(image_bgr):
    """Convert BGR to RGB tensor."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(img_rgb)

def compute_orientation(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    angle = rect[2]

    if rect[1][0] < rect[1][1]:
        angle = 90 + angle
    else:
        angle = -angle

    return box, angle

def run_webcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    cap = cv2.VideoCapture(0)  # Use your webcam (0 is default)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print("📷 Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Resize if needed
        frame_input = cv2.resize(frame, None, fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)

        # Preprocess
        tensor = preprocess_image(frame_input).to(device)

        with torch.no_grad():
            output = model([tensor])[0]

        scores = output["scores"].cpu().numpy()
        keep = scores >= CONFIDENCE_THRESH
        boxes = output["boxes"].cpu().numpy()[keep]
        masks = output["masks"].cpu().numpy()[keep]

        for box, mask_prob in zip(boxes, masks):
            mask = (mask_prob[0] > MASK_THRESH).astype(np.uint8) * 255
            rot_box, angle = compute_orientation(mask)
            if rot_box is None:
                continue

            # Draw rotated bounding box
            cv2.drawContours(frame_input, [rot_box], 0, (0, 255, 0), 2)

            # Axis-aligned bounding box (optional)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_input, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Draw angle label
            label = f"{angle:.1f}°"
            cv2.putText(
                frame_input, label,
                (rot_box[0][0], rot_box[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # Show frame
        cv2.imshow("📐 Degree Detection (Press Q to quit)", frame_input)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()
