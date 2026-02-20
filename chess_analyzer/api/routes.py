from flask import request, jsonify, current_app as app
from config import MODEL_PATH, TEMPLATE_PATH
from chess_analyzer.vision.detector import TemplateBoardDetector
from chess_analyzer.ml.predictor import PiecePredictor
from chess_analyzer.services.analysis_service import ChessAnalysisService

# --- SINGLETON INSTANCES ---
board_detector = TemplateBoardDetector(template_image_path=TEMPLATE_PATH)
piece_predictor = PiecePredictor(model_path=MODEL_PATH)
analysis_service = ChessAnalysisService(detector=board_detector, predictor=piece_predictor)

@app.route('/api/v1/analyze-board', methods=['POST'])
def analyze_board_endpoint():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided."}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        include_cropped_image = request.args.get("include_cropped_image", "true").strip().lower() not in {"0", "false", "no"}
        orientation = request.args.get("orientation", request.form.get("orientation", "White"))
        
        result, error_msg = analysis_service.analyze_image(
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
