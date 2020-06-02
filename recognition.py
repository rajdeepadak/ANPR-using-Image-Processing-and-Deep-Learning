import cv2, os
import numpy as np
from skimage import morphology
from char_model import get_model, predict_char
import time

def detect_chars(morph_img, original_image):
    img = original_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------ Find Contours ------------------------------

    contours, hierarchy = cv2.findContours(morph_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours,
                             key=lambda contour: cv2.boundingRect(contour)[0])

    # ---------------------------------------------------------------------------

    # ------------------------- Draw Bounding Rectangles ------------------------

    number = ''
    for i, ctr in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(ctr)

        roi = gray[y:y + h, x:x + w]
        area = w * h
        perimeter = 2 * (w + h)
        aspect_ratio = w / h

        if 350 < area < 2800 and 80 < perimeter < 475 and 0.15 < aspect_ratio < 2.2 and 10<w < 100 and 20<h < 70:
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("cropped plate", original_image)
            cv2.imshow("-", morph_img)
            cv2.imshow('rect', rect)

            crop_img = gray[int(y-0.05*h):int(y + 1.05*h), int(x-0.05*w):int(x + 1.05*w)]
            crop_img = cv2.adaptiveThreshold(crop_img, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
            if crop_img is None:
                continue
            crop_img = cv2.resize(crop_img,(28,28))
            #-------Recognize the detected character---------------
            char = predict_char(crop_img)
            #append characters to string
            number += char

            # print('Predicted character:',char)
            # cv2.imshow("Cropped Characters", crop_img)
            # cv2.waitKey(0)
    return number

    
def preprocess(original_image):
    # Resize Original Image
    img = cv2.resize(original_image, (350, 100))
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform Bilateral Filtering
    b_img = cv2.bilateralFilter(gray, 9, 70, 70)
    # Perform Canny Edge Detection
    ce_img = cv2.Canny(b_img, 10, 130)

    return img, ce_img

def apply_morphology(img):
    morph_img = img.copy()

    """____Redundant Lines Removal____________________________________________"""

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    remove_horizontal = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=8)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(morph_img, [c], -1, (0, 0, 0), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    remove_vertical = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, vertical_kernel, iterations=8)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(morph_img, [c], -1, (0, 0, 0), 5)

    """____Blob Removal_______________________________________________________"""

    br_img = morphology.remove_small_objects(morph_img.astype(bool), min_size=50,
                                             connectivity=3).astype(int)
    mask_x, mask_y = np.where(br_img == 0)
    morph_img[mask_x, mask_y] = 0

    """_______________________________________________________________________"""


    return morph_img



# for subdir, dirs, files in os.walk('E:/License-Plate-Recognition-System/Cropped/'):
#     for filename in files:
#         filepath = subdir + os.sep + filename

#         if filepath.endswith(".png"):
#             start = time.time()

#             # ------------------------------ Pre-Processing -----------------------------
#             # Read Original Image
#             img = cv2.imread(filepath)
            
#             resized_img, canny_img = preprocess(img)
            
#             # ---------------------------------------------------------------------------

#             # -------------------------------- Filtering --------------------------------

#             morph_img = apply_morphology(canny_img)

#             number = detect_chars(morph_img, resized_img)
#             end = time.time()
#             print('Time taken:',end - start)
#             print('Predicted number:',number)

#             cv2.waitKey(0)
#             cv2.destroyAllWindows

