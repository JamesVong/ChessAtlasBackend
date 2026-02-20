import cv2
import numpy as np

class TemplateBoardDetector:
    def __init__(self, template_image_path):
        template = cv2.imread(template_image_path)
        if template is None:
            raise FileNotFoundError(f"Template image not found at {template_image_path}")
        self.template_image = template
        self.template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        self.template_height, self.template_width = self.template_gray.shape[:2]
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
        ds_factor = 0.5
        frame_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Improve speed by downscaling frames
        frame_small = cv2.resize(
            frame_gray_full,
            None,
            fx=ds_factor,
            fy=ds_factor,
            interpolation=cv2.INTER_AREA,
        )
        template_gray_small = cv2.resize(
            self.template_gray,
            None,
            fx=ds_factor,
            fy=ds_factor,
            interpolation=cv2.INTER_AREA,
        )
        
        found = self.match_template(frame_small, template_gray_small, self.COARSE_SCALE_RANGE)
        if not found:
            return None

        _, coarse_max_loc, coarse_best_scale = found

        # Improve accuracy by performing fine scaling range
        fine_start = max(coarse_best_scale - 0.01, 0.01)
        fine_end = coarse_best_scale + 0.012
        fine_scale_range = np.arange(fine_start, fine_end, 0.002)
        fine_found = self.match_template(frame_gray_full, self.template_gray, fine_scale_range)
        
        if fine_found:
            _, max_loc, best_scale = fine_found
        else:
            max_loc = (int(coarse_max_loc[0] / ds_factor), int(coarse_max_loc[1] / ds_factor))
            best_scale = coarse_best_scale

        top_left = max_loc
        board_width = int(self.template_width * best_scale)
        board_height = int(self.template_height * best_scale)
        bottom_right = (top_left[0] + board_width, top_left[1] + board_height)
        
        return top_left, bottom_right
