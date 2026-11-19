# Degree Recognition

Real-time webcam tool that runs a pretrained Mask R-CNN (torchvision, COCO weights) on each frame. For every detected object it takes the instance mask, fits a minimum-area rotated rectangle, and draws that rectangle with an estimated orientation angle in degrees. It does not classify objects; it reports the angle of whatever COCO object is detected.

## Features

- Instance segmentation with `maskrcnn_resnet50_fpn`
- Rotated bounding box per object (`cv2.minAreaRect`) with an angle label
- Axis-aligned bounding box drawn alongside the rotated box
- Uses CUDA if available, otherwise CPU
- Confidence, mask threshold and frame scale set at the top of `test.py`

## Project structure

- `test.py`: the whole program. Contains `load_model`, `preprocess_image`, `compute_orientation` and `run_webcam`. Opens webcam index 0; press `q` to quit.
- `annotated.jpg`, `annotated_bottle.jpg`, `annotated_output.jpg`: sample output frames.

## Requirements

Imports used: `cv2`, `numpy`, `torch`, `torchvision`.

```
pip install opencv-python numpy torch torchvision
```

## Installation

```
git clone https://github.com/Yaldram69/Degree-Recognition.git
cd Degree-Recognition
pip install opencv-python numpy torch torchvision
```

## Usage

Run from the repository folder:

```
python test.py
```

The script takes no arguments. It reads from the default webcam (index 0), so no image or video path is needed. On the first run torchvision downloads the COCO Mask R-CNN weights, which requires internet access.

Tunable constants at the top of `test.py`:

- `CONFIDENCE_THRESH` (default 0.5): minimum detection score
- `MASK_THRESH` (default 0.5): mask binarization threshold
- `FRAME_DOWNSCALE` (default 1.0): set below 1.0 to speed up processing

## Notes

- Model weights are not included in the repository; they are downloaded by torchvision on first run.
- `pretrained=True` is the older torchvision API and may emit a deprecation warning; newer versions use `weights=`.
- `compute_orientation` uses `np.int0`, which was removed in NumPy 2.0. On NumPy 2.x replace it with `np.intp`.
- All COCO classes are processed; there is no class filter.
- The angle is a heuristic from the rotated rectangle and has not been calibrated or evaluated against ground truth.
- No training code is included.
