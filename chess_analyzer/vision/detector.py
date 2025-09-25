import cv2
import numpy as np

class TemplateBoardDetector:
    def __init__(self, template_image_path):
        template = cv2.imread(template_image_path)
        if template is None:
            raise FileNotFoundError(f"Template image not found at {template_image_path}")
        self.template_image = template
        self.template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        self.COARSE_SCALE_RANGE = np.linspace(0.05, 0.7, 50)
        self.THRESHOLD = 0.4

    def match_template(self, test_gray, template_gray, scale_range):
        found = None
        for scale in scale_range:
            resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if resized_template.shape[0] > test_gray.shape[0] or resized_template.shape[1] > test_gray.shape[1]: continue
            result = cv2.matchTemplate(test_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > self.THRESHOLD and (found is None or max_val > found[0]):
                found = (max_val, max_loc, scale)
        return found
    
    def detect(self, image):
        # ... (rest of the detect logic from your class)
        frame_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = self.match_template(frame_gray_full, self.template_gray, self.COARSE_SCALE_RANGE) # Simplified for clarity
        if not found:
            return None
        max_val, max_loc, best_scale = found
        h, w = self.template_gray.shape
        best_h = int(h * best_scale)
        best_w = int(w * best_scale)
        top_left = max_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        return top_left, bottom_right