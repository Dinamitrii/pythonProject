from imutils.video import FPS
import numpy as np
import imutils
import cv2

use_gpu = True
live_video = True
confidence_level = 0.5
fps = FPS().start()
ret = True
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "dining table", "dog", "horse",
           "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv-monitor"]

id_xternal = int()

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("ssd_files/MobileNetSSD_deploy.prototxt", "ssd_files/MobileNetSSD_deploy.caffemodel")

if use_gpu:
    print("[INFO] setting preferable backend to OpenCV and preferable target to OpenCL...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)

print("[INFO] accessing video stream...")

if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture("test.mp4")

while ret:
    ret, frame = vs.read()
    if ret:
        frame = imutils.resize(frame, width=480)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (640, 480), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_level:
                id_xternal = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[id_xternal], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[id_xternal], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id_xternal], 2)

        cv2.imshow("Live detection", frame)

        if cv2.waitKey(1) == 27:
            break

        fps.update()

fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
