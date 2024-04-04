# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# Load image
image = cv2.imread('mona.jpg')

# Test image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Test image
# plt.imshow(gray_image, cmap='gray')

# Function for face detecting
def detect_face(I):
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(I, 1.3, 5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(I, (x, y), (x+w, y+h), (0, 255, 255), 1)
    
    return I

# Detect faces in the image
image_with_face_detected = detect_face(image.copy())

# Test image
# cv2_imshow(image_with_face_detected) 

# Apply Gaussian blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

# Test blurred image
# plt.imshow(blurred_image, cmap='gray')

# Apply 3-channel to the grayscale image
colorized_image = cv2.applyColorMap(gray_image, cv2.COLOR_GRAY2BGR)

# Test image
# cv2_imshow(colorized_image)

def create_face_mask(image):
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    
    # Create a blank mask with the same shape as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw rectangles around the detected faces and create the mask
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255), -1)
    
    return mask


# Convert the original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask of the face region
mask = create_face_mask(image)

# Preserve the face area from the original image
face_region = cv2.bitwise_and(image, image, mask=mask)

# Invert the mask to select the non-face region
mask_inv = cv2.bitwise_not(mask)

# Convert the grayscale image to 3 channels
gray_image_3_channels = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Combine the face region and the grayscale image
result_image = cv2.bitwise_or(face_region, gray_image_3_channels, mask=mask_inv)

# Add the face region to the resulting image
result_image_with_face = cv2.add(result_image, face_region)

# Create subplots with adjusted figure size
plt.figure(figsize=(18, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')

# Mona Lisa with detected face
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image_with_face_detected, cv2.COLOR_BGR2RGB))
plt.title('Mona Lisa with detected face')

# Resulting image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result_image_with_face, cv2.COLOR_BGR2RGB))
plt.title('Gray outside the face of Mona Lisa')

# Show subplots
plt.show()


