import cv2
import numpy as np
import time

# Open a video capture (0 for the default camera)
camPath = '/home/madsr2d2/test.mp4'
cap = cv2.VideoCapture(camPath, cv2.CAP_FFMPEG)
refFrame = None
T = 50

while True:

    start = time.time()
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print('Error reading frame')
        break

    # Get the reference frame and convert it to grayscale and float
    if refFrame is None:
        refFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        refFrame = refFrame.astype(np.float32)
        continue

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the image to float
    grayImgAsFloat = grayImage.astype(np.float32)

    # Calculate the absolute difference between the current frame and the reference frame
    diff = np.abs(grayImgAsFloat - refFrame)

    # Generate binary image based on the threshold T
    _, mask = cv2.threshold(diff, T, 255, cv2.THRESH_BINARY)
    print(mask)

    # covert image to uint8
    grayImageUint8 = grayImgAsFloat.astype(np.uint8)

    stop = time.time()
    fps = 1/(stop-start)
    cv2.putText(grayImageUint8, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', grayImageUint8)
    cv2.imshow('Mask', mask)

    # Wait for 1 millisecond for a key event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Break the loop if 'q' is pressed
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
