import os
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import imutils

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    # Use correct size for SSD model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32, verbose=0)
    else:
        preds = []

    return (locs, preds)

def main():
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    maskModelPath = "mask_detector.h5"

    for fpath, message in [
        (prototxtPath, "Face detector prototxt not found!"),
        (weightsPath, "Face detector weights not found!"),
        (maskModelPath, "Mask detector model not found!")
    ]:
        if not os.path.isfile(fpath):
            print(f"[ERROR] {message} Expected at: {os.path.abspath(fpath)}")
            return

    print("[INFO] Loading face detector model...")
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] Loading face mask detector model...")
    maskNet = load_model(maskModelPath)

    print("[INFO] Starting video stream. Press 'q' to quit.")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    try:
        while True:
            frame = vs.read()
            if frame is None:
                print("[WARNING] Frame not read from webcam.")
                break

            frame = imutils.resize(frame, width=600)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # Label and color logic
                if mask > withoutMask:
                    label = "Mask"
                    color = (0, 255, 0)  # green
                    label_text = f"Mask: {mask*100:.2f}%"
                else:
                    label = "No Mask"
                    color = (0, 0, 255)  # red
                    label_text = f"No Mask: {withoutMask*100:.2f}%"

                # Draw label and rectangle
                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow("Face Mask Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        print("[INFO] Cleaning up...")
        cv2.destroyAllWindows()
        vs.stop()

if __name__ == "__main__":
    main()