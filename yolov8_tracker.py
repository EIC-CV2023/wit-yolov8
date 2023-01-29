from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from ultralytics.SORT import *


model = YOLO("cokebest-fs-seg.pt")

# For coco
# with open("ultralytics/yolo/data/datasets/coco8-seg.yaml", "r") as stream:
#     try:
#         datasets = yaml.safe_load(stream)
#         datasets_names = datasets['names']
#     except:
#         print("No file found")
#         datasets_names = ""

datasets_names = {0: "coke"}


def init_tracker():
    global tracker
    sort_max_age = 10
    sort_min_hits = 5
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits,
                   iou_threshold=sort_iou_thresh)


def draw_box(img, bbox, id=None, label=None):
    x1, y1, x2, y2 = bbox
    # print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), rand_color_list[id % 20], 3)
    cv2.putText(img, f"{id}:{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1, rand_color_list[id % 20], 2)
    return img


start = time.time()
cap = cv2.VideoCapture(0)

init_tracker()
rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # print("fps: " + str(round(1 / (time.time() - start), 2)))
    start = time.time()
    frame2 = np.copy(frame)

    results = model.predict(source=frame, conf=0.5, show=True)[0]
    if results.boxes:
        # print(f"DETECT {len(results.boxes)}")
        output = dict()
        dets_to_sort = np.empty((0, 6))

        for i, obj in enumerate(results.boxes):
            x1, y1, x2, y2, conf, cls = obj.data.cpu().detach().numpy()[0]
            name = datasets_names[int(cls)] if datasets_names else 'unknown'

            output[i] = [name, x1, y1, x2, y2]

            dets_to_sort = np.vstack((dets_to_sort,
                                      np.array([x1, y1, x2, y2, conf, cls])))
        # print(dets_to_sort)

        tracked_dets = tracker.update(dets_to_sort)
        # print(tracked_dets)
        for tk in tracked_dets:
            bbox_xyxy = [int(p) for p in tk[:4]]
            id = int(tk[8])
            name = datasets_names[tk[4]]
            # print(bbox_xyxy, id, name)

            draw_box(frame2, bbox_xyxy, id, name)
            res.append([name, bbox_xyxy])

    # print(output)
    print(res)

    cv2.imshow("frame", frame2)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
