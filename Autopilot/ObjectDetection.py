import cv2
import numpy as np


class YOLOv3_net():
    """
    YOLO_net class for object detection

    TODO: Replace this network with Tensorflow based YOLOv4 algorithm
    """
    def __init__(self):
        self.Net = cv2.dnn.readNet("YOLOv3/yolov3-tiny.weights", "YOLOv3/yolov3-tiny.cfg")

        self.classes = []
        with open("YOLOv3/yolo.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.Net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.Net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        """
        Make inference on the object in the given frame.
        :param frame: The input frame containing objects that we want to detect.
        :return: A tuple of lists required to draw bounding boxes. This consists of 'class_ids', 'confidences', 'boxes', 'indices'
        """
        height, width, channel = frame.shape

        # inputs to YOLO network
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0),
                                     True, crop=False)
        self.Net.setInput(blob)
        outputs = self.Net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:

            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object is detected
                    # The coordinate of boxes are normalized to be in between [0, 1]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    dw = int(detection[2] * width)
                    dh = int(detection[3] * height)

                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


        return class_ids, confidences, boxes

    def draw_boxes(self, frame, class_ids, confidences, boxes):
        """
        Draw bounding boxes with given data
        :param frame: A frame to draw bounding boxes on
        :param class_ids: A list containing class IDs of detected objects
        :param confidences: A list containing confidence scores of each detection
        :param boxes: A list of lists containing position, size information of bounding boxes
        :return: An annotated frame
        """

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

        for i in range(len(boxes)):
            if i in indices:

                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                score = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

        return frame
