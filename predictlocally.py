import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fire_detection_model.h5')

# Load the video
cap = cv2.VideoCapture('fire.mp4')

# Define a function to preprocess each frame of the video
def preprocess_frame(frame):
    # Resize the frame to the input size of the model (224x224)
    resized_frame = cv2.resize(frame, (224, 224))
    # Convert the image to a format that can be used by the model (float32 array)
    input_image = resized_frame.astype('float32') / 255.0
    # Add an extra dimension to the input to match the input shape of the model (batch size of 1)
    input_image = tf.expand_dims(input_image, axis=0)
    return input_image

# Loop through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_image = preprocess_frame(frame)

    # Run the model on the input image
    prediction = model.predict(input_image)[0][0]

    # Print the prediction score for debugging purposes
    print(f"Prediction score: {prediction}")

    # If the prediction is greater than a threshold value (e.g. 0.5), find the location of the fire and draw a rectangle around it
    if prediction > 0.5:
        # Convert the input image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to the grayscale image to remove noise
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Apply adaptive thresholding to the blurred image to segment the fire from the background
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a rectangle around each contour that corresponds to the location of the fire
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
