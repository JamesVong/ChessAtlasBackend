# Chess Atlas — Backend

REST API that accepts a chess board image and returns the detected position as a FEN string. Used by [chess-atlas.com](https://chess-atlas.com).

## How it works

1. **Board detection** — A YOLOv8 model (`best_320.onnx`, 320×320 input) locates the chessboard in the image and returns a bounding box.
2. **Square splitting** — The crop is divided into 64 squares. Boards narrower than 896 px are upscaled first (INTER_CUBIC) so each raw square is at least 112 px before the final resize to 224×224.
3. **Piece classification** — A fine-tuned MobileNetV3-Small (`chess_atlas_v1.onnx`) classifies each square into one of 13 classes (12 piece types + Empty). Inference runs in batches of 8 to keep memory bounded.
4. **FEN assembly** — Square labels are assembled into a board-position FEN string and returned in the response.

## API

### `POST /api/v1/analyze-board`

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | JPEG or PNG, max 8 MB |
| `orientation` | string | `White` (default) or `Black` — which side is at the bottom |

Query parameter `include_cropped_image=false` skips the base64 board crop in the response (faster).

**Response (success)**

```json
{
  "status": "success",
  "data": {
    "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
    "cropped_image": "data:image/jpeg;base64,..."
  }
}
```

**Response (error)**

```json
{
  "status": "error",
  "message": "Could not find a chessboard in the image."
}
```

| Status | Meaning |
|---|---|
| 200 | Success |
| 400 | No image file in request |
| 413 | Image exceeds size limit |
| 422 | Board not detected, or invalid orientation |
| 500 | Unexpected server error |

## Models

| File | Purpose | Input |
|---|---|---|
| `models/best_320.onnx` | YOLOv8 board detector | 1×3×320×320 float32 |
| `models/chess_atlas_v1.onnx` + `.onnx.data` | MobileNetV3-Small piece classifier | N×3×224×224 float32 |

The piece classifier was trained on chess board screenshots with heavy augmentation (JPEG compression simulation, streamer arrow/highlight overlays, colour jitter, random erasing) and a `WeightedRandomSampler` that oversamples rare piece classes. It achieves **0.9996 macro-F1** on the validation set and **0.9944** on the held-out test set across 13 classes.

Normalisation stats (computed from the train split, RGB):

```
mean = [0.7676, 0.6587, 0.5281]
std  = [0.1998, 0.2000, 0.1923]
```

## Project structure

```
chess_analyzer/
  api/
    routes.py          # Flask route, lazy model loading
  ml/
    predictor.py       # ONNX inference, batched preprocessing
  services/
    analysis_service.py  # Orchestrates detection → split → predict → FEN
  vision/
    detector.py        # YoloBoardDetector (letterbox, NMS)
    preprocessing.py   # Board crop, upscale guard, square division
models/
  best_320.onnx        # Board detector
  chess_atlas_v1.onnx  # Piece classifier
  chess_atlas_v1.onnx.data
config.py              # All paths, sizes, label maps
chess_training.ipynb   # Full training pipeline (PyTorch)
Dockerfile
requirements.txt
```

## Running locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

The server starts on `http://localhost:5001`. Models are loaded lazily on the first request.

## Deployment

Runs as a single-worker gunicorn process inside Docker on Render:

```
gunicorn --workers 1 --threads 1 --max-requests 40 --timeout 120 run:app
```

The container memory limit is 512 MB. Key settings that keep it within budget:
- TensorFlow removed from dependencies (saved ~300 MB RSS)
- Inference batch size capped at 8 (bounds MobileNetV3 activation memory)
- `OMP_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2` to suppress glibc arena bloat

## Training

See `chess_training.ipynb`. Requires a dataset of labelled chess square images and a `split_assignments.csv`. The notebook exports both TorchScript and ONNX checkpoints and verifies numerical consistency between them.
