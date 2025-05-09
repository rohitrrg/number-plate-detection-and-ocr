import cv2
import numpy as np
from tensorflow.keras.models import load_model



class LiscencePlateExtractor:

    def __init__(self):
        # Load the Haar Cascade XML file for license plate detection
        self.plate_cascade = cv2.CascadeClassifier('/Users/somdipsen/ML_Lab/IPML/NumberplateDetector2/indian_license_plate.xml')
        self.model = load_model('/Users/somdipsen/ML_Lab/IPML/NumberplateDetector2/model.h5')
        self.track = []
        self.dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, c in enumerate(characters):
            self.dic[i] = c

    def detect_plates(self, gray_image):
        
        # Detect license plates in the image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        plate_rects = self.plate_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=7)
        
        # Check if any plates are detected
        if len(plate_rects) == 0:
            print("No license plates detected.")
        else:
            print(f"{len(plate_rects)} license plate(s) detected.")
        
        plates = []
        # Iterate through detected plates
        for (x, y, w, h) in plate_rects:
            # Extract the region of interest (ROI) for the plate
            plate = gray_image[y:y+h, x:x+w]
            plates.append(plate)
        
        return plates

    def detect_chars(self, plate):
        resized = cv2.resize(plate, (333, 75))
        # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)
        _, binary = cv2.threshold(denoised, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=2)

        # Add white borders to assist contour detection
        binary[0:3, :] = 255
        binary[:, 0:3] = 255
        binary[-3:, :] = 255
        binary[:, -3:] = 255

        # Estimate dimensions for valid character contours
        height, width = binary.shape
        dimensions = [
            height / 9,    # min width
            height / 2,    # max width
            width / 10,    # min height
            2 * width / 3  # max height
        ]
        lower_w, upper_w, lower_h, upper_h = dimensions

        # Find contours in the image
        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        x_coords = []
        char_images = []

        # Make a debug image to draw boxes
        debug_image = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if lower_w < w < upper_w and lower_h < h < upper_h:
                x_coords.append(x)

                # Extract character
                char_img = binary[y:y+h, x:x+w]
                char_img = cv2.resize(char_img, (20, 40))
                char_img = cv2.subtract(255, char_img)  # Invert

                # Pad character to 24x44
                padded = np.zeros((44, 24))
                padded[2:42, 2:22] = char_img

                char_images.append(padded)
        
        # Sort characters from left to right
        sorted_chars = [char_images[i] for i in sorted(range(len(x_coords)), key=lambda k: x_coords[k])]

        return sorted_chars
    
    def fix_dimension(self,img):
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def show_results(self, char):
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,c in enumerate(characters):
            dic[i] = c

        output = []
        for i,ch in enumerate(char): #iterating over the characters
            img_ = cv2.resize(ch, (28,28))
            img = self.fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_probs = self.model.predict(img, verbose=0)[0] #predicting the class
            y_ = np.argmax(y_probs)
            character = dic[y_]
            output.append(character) #storing the result in a list

        plate_number = ''.join(output)

        return plate_number

    def extract(self, image):
        plates = self.detect_plates(image)
        result = []
        for plate in plates:
            sorted_chars = self.detect_chars(plate)
            if len(sorted_chars) >= 6:
                res = self.show_results(sorted_chars)
                result.append(res)
        return result