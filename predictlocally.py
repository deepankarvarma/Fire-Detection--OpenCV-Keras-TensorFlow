import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('smoke_detection_model.h5')

# Load the video
cap = cv2.VideoCapture('smoke.mp4')

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

    # If the prediction is greater than a threshold value (e.g. 0.1), draw a rectangle around the detected area
    if prediction > 0.1:
        cv2.rectangle(frame, (0, 0), (224, 224), (0, 0, 255), 2)

    # Show the resulting frame
    cv2.imshow('Smoke Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()