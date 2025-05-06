import cv2
from ultralytics import YOLO  # Make sure you have the right version installed

class LicensePlateDetector:
    def __init__(self):
        self.model = YOLO(r'./runs\detect\train4\weights\best.pt')  # Load trained YOLO model

    def detect_and_crop(self, image_input):
        # Read image if a path is given
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        results = self.model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        
        if len(boxes) == 0:
            return None  # No detections
        
        cropped_plates = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            cropped_plates.append(cropped)

        return cropped_plates

    def detect_and_crop_with_conf(self, image_input, conf_threshold=0.8):
        # Read image if a path is given
        
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        results = self.model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        if len(confs) == 0:
            return None  # No detections
        
        cropped_plates = []
        for box, conf in zip(boxes, confs):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                cropped = image[y1:y2, x1:x2]
                cropped_plates.append(cropped)

        return cropped_plates

    def detect_highest_conf_plate(self, image_input):
        # Read image if a path is given
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        results = self.model(image)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        if len(confs) == 0:
            return None  # No detections

        # Get index of highest confidence detection
        max_idx = confs.argmax()
        x1, y1, x2, y2 = map(int, boxes[max_idx])
        cropped_plate = image[y1:y2, x1:x2]
        
        return cropped_plate


