import cv2

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam is connected and ready!")

# Release the webcam
cap.release()
cv2.destroyAllWindows()
