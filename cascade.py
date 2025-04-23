import cv2

class LiscencePlateDetector:

    def __init__(self, single=True):
        self.model = cv2.CascadeClassifier('./saved_model/indian_license_plate.xml')
        self.single = single

    def detect_plate(self, image_input):
         # Read image if a path is given
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.model.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7)

        if len(plates)==0:
            return None

        if self.single:
            bbox = plates[0]
            (x, y, w, h) = bbox
            # Extract the region of interest (ROI) for the plate
            plate = image[y:y+h, x:x+w]
            return plate
        else:
            ps = []
            for bbox in plates:
                (x, y, w, h) = bbox
                # Extract the region of interest (ROI) for the plate
                plate = image[y:y+h, x:x+w]
                ps.append(plate)
            return ps