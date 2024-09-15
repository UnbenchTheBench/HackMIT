import cv2
import numpy as np
import threading
import queue
import pyttsx3

class CameraThread(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.cap = cv2.VideoCapture(0)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (640, 480))
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                continue
            
            # Put frame in queue if queue size is below limit
            if self.frame_queue.qsize() < 10:
                self.frame_queue.put(frame)
            
            self.out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        self.out.release()

    def stop(self):
        self.running = False

class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True

        # Initialize the TTS engine
        self.tts_engine = pyttsx3.init()

        # Load YOLO
        try:
            self.net = cv2.dnn.readNet('model/yolov4.weights', 'model/yolov4.cfg')
            self.layer_names = self.net.getLayerNames()
            out_layer_indices = self.net.getUnconnectedOutLayers()
            if isinstance(out_layer_indices, np.ndarray):
                out_layer_indices = out_layer_indices.flatten()
            self.output_layers = [self.layer_names[i - 1] for i in out_layer_indices]

            # Load class labels
            with open('model/coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.running = False

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.objectDetection(frame)

                # Put results in queue for main thread
                results_queue.put(frame)

    def objectDetection(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Loop over each output layer's results
        for out in outs:
            for detection in out:
                if len(detection) > 5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    confidence = detection[4]

                    if confidence > 0.5:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        class_confidence = scores[class_id]

                        if class_confidence > 0.5:
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Speak the detected object's name
                self.speak(label)

    def speak(self, text):
        """Speak out the detected object's name using pyttsx3"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def stop(self):
        self.running = False

def main():
    frame_queue = queue.Queue()
    global results_queue
    results_queue = queue.Queue()

    camera_thread = CameraThread(frame_queue)
    processing_thread = ProcessingThread(frame_queue)

    camera_thread.start()
    processing_thread.start()

    while True:
        if not results_queue.empty():
            frame = results_queue.get()
            cv2.imshow('Object Detection', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            camera_thread.stop()
            processing_thread.stop()
            break

    camera_thread.join()
    processing_thread.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
