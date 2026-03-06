from flask import request, jsonify, current_app as app


def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1


# Lazy-load models on first request to avoid blocking worker boot.
_analysis_service = None

def _get_service():
    global _analysis_service
    if _analysis_service is None:
        from config import MODEL_PATH, DETECTOR_MODEL_PATH, DETECTOR_INPUT_SIZE
        from chess_analyzer.vision.detector import YoloBoardDetector
        from chess_analyzer.ml.predictor import PiecePredictor
        from chess_analyzer.services.analysis_service import ChessAnalysisService

        print(f"[MEM] before loading detector: {_rss_mb():.1f} MB", flush=True)
        board_detector = YoloBoardDetector(model_path=DETECTOR_MODEL_PATH, input_size=DETECTOR_INPUT_SIZE)
        print(f"[MEM] after loading detector: {_rss_mb():.1f} MB", flush=True)
        piece_predictor = PiecePredictor(model_path=MODEL_PATH)
        _analysis_service = ChessAnalysisService(detector=board_detector, predictor=piece_predictor)
        print(f"[MEM] all models loaded: {_rss_mb():.1f} MB", flush=True)
    return _analysis_service

@app.route('/api/v1/analyze-board', methods=['POST'])
def analyze_board_endpoint():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        include_cropped_image = request.args.get("include_cropped_image", "true").strip().lower() not in {"0", "false", "no"}
        orientation = request.args.get("orientation", request.form.get("orientation", "White"))

        result, error_msg = _get_service().analyze_image(
            image_bytes,
            include_cropped_image=include_cropped_image,
            orientation=orientation,
        )

        if error_msg:
            return jsonify({"status": "error", "message": error_msg}), 422

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"status": "error", "message": "An internal server error occurred."}), 500
