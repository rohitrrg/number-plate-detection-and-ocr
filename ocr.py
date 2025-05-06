import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class CNN:
    def __init__(self):
        self.model = load_model('./saved_model/license_plate_model.h5')
    
    def fix_dimension(self,img):
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def predict(self, chars):
        output = []
        for i,ch in enumerate(chars): #iterating over the characters
            img_ = cv2.resize(ch, (28,28))
            img = self.fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_probs = self.model.predict(img, verbose=0)[0] #predicting the class
            y_ = np.argmax(y_probs)
            character = self.dic[y_] #
            output.append(character) #storing the result in a list

        return ''.join(output)

class LiscencePlateExtractor:

    def __init__(self):

        self.model = load_model('./saved_model/license_plate_model.h5')

        self.dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i,c in enumerate(characters):
            self.dic[i] = c        

    def process(self, plate):

        # self.track.append(plate)

        resized = cv2.resize(plate, (333, 75))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        self.track.append(gray)

        #denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self.track.append(binary)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # kernel = np.ones((3, 3), np.uint8)
        # binary = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=2)

        binary = cv2.erode(binary, (3,3))
        binary = cv2.dilate(binary, (3,3))

        # # Add white borders to assist contour detection
        # binary[0:3, :] = 255
        # binary[:, 0:3] = 255
        # binary[-3:, :] = 255
        # binary[:, -3:] = 255

        binary[0:12,:] = 255
        binary[:,0:12] = 255
        binary[65:75,:] = 255
        binary[:,323:333] = 255

        # h, w = binary.shape
        # binary = binary[12:h-10, 12:w-10]

        self.track.append(binary)

        return binary

    def detect_chars(self, plate):

        # Estimate dimensions for valid character contours
        width, height = plate.shape
        print('Width:', width, 'Height:',height)
        dimensions = [
            width / 10,    # min width
            width / 1.1,    # max width
            height / 10,    # min height
            2*height / 3  # max height
        ]
        lower_w, upper_w, lower_h, upper_h = dimensions

        # Find contours in the image
        contours, _ = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        print("NUMBER OF CONTOURS FOUND:", len(contours))
        x_coords = []
        char_images = []

        # Make a debug image to draw boxes
        debug_image1 = cv2.cvtColor(plate.copy(), cv2.COLOR_GRAY2BGR)
        debug_image2 = debug_image1.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if lower_w < w < upper_w and lower_h < h < upper_h:
                x_coords.append(x)

                # Extract character
                char_img = plate[y:y+h, x:x+w]
                char_img = cv2.resize(char_img, (20, 40))
                char_img = cv2.subtract(255, char_img)  # Invert

                # Pad character to 24x44
                padded = np.zeros((44, 24))
                padded[2:42, 2:22] = char_img

                char_images.append(padded)

                cv2.rectangle(debug_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.track.append(debug_image2)
        self.track.append(debug_image1)

        print("NUMBER OF SHORTLISTED CONTOURS:", len(char_images))
        # Sort characters from left to right
        sorted_chars = [char_images[i] for i in sorted(range(len(x_coords)), key=lambda k: x_coords[k])]
 
        if len(sorted_chars)>0:
            # Show segmented characters
            print('LENGTH_:',len(sorted_chars))
            fig, axs = plt.subplots(1, len(sorted_chars), figsize=(15, 3))
            for i, char in enumerate(sorted_chars):
                axs[i].imshow(char, cmap='gray')
                axs[i].axis('off')
            plt.suptitle("Segmented Characters")
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)

            # Load buffer into PIL Image
            img_pil = Image.open(buf)
            self.track.append(img_pil)

        return sorted_chars

    def fix_dimension(self,img):
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def predict(self, chars):
        output = []
        for i,ch in enumerate(chars): #iterating over the characters
            img_ = cv2.resize(ch, (28,28))
            img = self.fix_dimension(img_)
            img = img.reshape(1,28,28,3) #preparing image for the model
            y_probs = self.model.predict(img, verbose=0)[0] #predicting the class
            y_ = np.argmax(y_probs)
            character = self.dic[y_] #
            output.append(character) #storing the result in a list

        return ''.join(output)

    def extract(self, plate):
        self.track = []

        binary_img = self.process(plate)
        sorted_chars = self.detect_chars(binary_img)

        return sorted_chars, self.track