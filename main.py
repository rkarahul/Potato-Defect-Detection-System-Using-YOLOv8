import base64
import io
import time
import torch
import numpy as np
import cv2
import asyncio
import traceback
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO
import uvicorn
import math
import supervision as sv

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the YOLOv8 models and move to device
modeldefect = YOLO(r"best_defect.pt").to(device)  # Defect model
modelgood = YOLO(r"best_good.pt").to(device)      # Good potato model

# Initialize FastAPI
app = FastAPI()

# Image request schema
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    camcode: int  # Camera code

def image_to_base64(image: np.ndarray) -> str:
    """ Convert OpenCV image to base64 string """
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def calculate_polygon_area(polygon):
    """ Compute the area of a polygon in mm² """
    ppm = 300 / 640  # Pixels per mm
    area_pixels = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + [polygon[0]])))
    area_mm2 = round(area_pixels * (ppm ** 2), 2)
    print(f"Area in pixels: {area_pixels}, Area in mm²: {area_mm2}")
    return area_mm2

def scalecord(cord):
    """ Scale coordinates to a 300x300 field """
    try:
        x, y = cord
        x = 640 - x  # Flipped
        return [int((x / 640) * 292), int((y / 640) * 314)]
    except:
        return cord

def scaleWH(cord):
    """ Scale width and height to a 300x300 field """
    try:
        x, y = cord
        return [int((x / 640) * 292), int((y / 640) * 314)]
    except:
        return cord

def map_defects_to_potatoes(potato, defect):
    """ Map defects to potatoes based on center inclusion """
    def is_inside(center, box):
        x, y = center
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    resdect = {
        "Potato": [],
        "PotatoArea": [],
        "Potato_WH": [],
        "PotatoCenter": [],
        "TrackingID": [],  # Tracking IDs for good potatoes, -1 for defective
        "defectname": [],
        "DefectArea": [],
        "DefectCord": []
    }

    for p_box, (p_center, p_hw, p_area, p_track_id) in potato.items():
        resdect["Potato"].append(list(p_box))
        resdect["PotatoArea"].append(round(p_area, 2))
        resdect["Potato_WH"].append([p_hw[0], p_hw[1]])
        resdect["PotatoCenter"].append([p_center[0], p_center[1]])
        resdect["TrackingID"].append(int(p_track_id) if p_track_id is not None else -1)

        matched_names = []
        matched_areas = []
        matched_coords = []

        for d_box, (d_center, d_name, d_area) in defect.items():
            if is_inside(d_center, p_box):
                matched_names.append(d_name.lower())
                matched_areas.append(round(d_area, 2))
                matched_coords.append(list(d_box))

        resdect["defectname"].append(matched_names)
        resdect["DefectArea"].append(matched_areas)
        resdect["DefectCord"].append(matched_coords)

    return resdect

def remove_boundary_potatoes(data, img_width=None, img_height=None):
    """ Remove potatoes touching image boundaries """
    keep_indices = []
    for idx, (x_min, y_min, x_max, y_max) in enumerate(data['Potato']):
        if x_min <= 2 or y_min <= 2:
            continue  # Touching left or top
        if img_width and x_max >= img_width:
            continue  # Touching right
        if img_height and y_max >= img_height:
            continue  # Touching bottom
        keep_indices.append(idx)

    filtered_data = {}
    for key, values in data.items():
        filtered_data[key] = [values[i] for i in keep_indices]
    
    return filtered_data 

def merge_good_defective_potatoes(data, proximity=10):
    """ Merge good and defective potatoes based on proximity """
    num_potatoes = len(data['Potato'])
    good_potatoes = set()

    for i in range(num_potatoes):
        for j in range(i+1, num_potatoes):
            box1 = data['Potato'][i]
            box2 = data['Potato'][j]

            box1_expand = [box1[0] - proximity, box1[1] - proximity, box1[2] + proximity, box1[3] + proximity]

            if (box1_expand[0] < box2[2] and box1_expand[2] > box2[0] and
                box1_expand[1] < box2[3] and box1_expand[3] > box2[1]):
                if (len(data['defectname'][i]) == 0 and len(data['defectname'][j]) > 0) or \
                   (len(data['defectname'][j]) == 0 and len(data['defectname'][i]) > 0):
                    good_potatoes.add(i)
                    good_potatoes.add(j)

    new_data = {key: [] for key in data}
    for idx in range(num_potatoes):
        if idx in good_potatoes:
            new_data['Potato'].append(data['Potato'][idx])
            new_data['PotatoArea'].append(data['PotatoArea'][idx])
            new_data['Potato_WH'].append(data['Potato_WH'][idx])
            new_data['PotatoCenter'].append(data['PotatoCenter'][idx])
            new_data['TrackingID'].append(data['TrackingID'][idx])
            new_data['defectname'].append([])
            new_data['DefectArea'].append([])
            new_data['DefectCord'].append([])
        else:
            new_data['Potato'].append(data['Potato'][idx])
            new_data['PotatoArea'].append(data['PotatoArea'][idx])
            new_data['Potato_WH'].append(data['Potato_WH'][idx])
            new_data['PotatoCenter'].append(data['PotatoCenter'][idx])
            new_data['TrackingID'].append(data['TrackingID'][idx])
            new_data['defectname'].append(data['defectname'][idx])
            new_data['DefectArea'].append(data['DefectArea'][idx])
            new_data['DefectCord'].append(data['DefectCord'][idx])

    return new_data

def get_center_and_size(x1, y1, x2, y2):
    """ Calculate center and size of a rectangle """
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2
    return ((center_x, center_y), (width, height))

async def detect_defects(frame, camcode):
    """ Run YOLO inference for both models and track only good potatoes """
    loop = asyncio.get_running_loop()
    print("Size of image: ", frame.shape)

    # Run predictions for both models on the specified device
    results_defect = await loop.run_in_executor(None, lambda: modeldefect.predict(frame, conf=0.6, iou=0.65, device=device))
    results_good = await loop.run_in_executor(None, lambda: modelgood.predict(frame, conf=0.6, iou=0.65, device=device))

    # Initialize containers
    potato = {}
    defect = {}
    defect_areas, potato_sizes, defect_names, scaled_coords = [], [], [], []
    plot_corr_POTATO, plot_corr_damage = [], []

    # Initialize tracker for this frame (good potatoes only)
    tracker = sv.ByteTrack()  # Reset tracker for single-frame IDs

    # Process good potato results with tracking
    boxes_good = results_good[0].boxes.xyxy.cpu().numpy()
    scores_good = results_good[0].boxes.conf.cpu().numpy()
    classes_good = results_good[0].boxes.cls.cpu().numpy()
    names_good = results_good[0].names

    # Prepare detections for ByteTrack (good potatoes)
    detections = sv.Detections(
        xyxy=boxes_good,
        confidence=scores_good,
        class_id=classes_good.astype(int)
    )

    # Update tracker with good potato detections
    detections = tracker.update_with_detections(detections)

    for box, score, cls, track_id in zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id):
        try:
            x_min, y_min, x_max, y_max = map(int, box)
            if x_min <= 2 or y_min <= 2 or x_max >= 636 or y_max >= 636:
                continue
            class_label = names_good[int(cls)]
            rect = get_center_and_size(x_min, y_min, x_max, y_max)

            scale_x = 0.45625  # mm per pixel
            scale_y = 0.49062  # mm per pixel
            width_px, height_px = rect[1]
            width_mm = width_px * scale_x
            height_mm = height_px * scale_y
            potato_size_mm = math.sqrt(width_mm**2 + height_mm**2)
            poly_area = round(potato_size_mm, 2)

            scaled_cord = scalecord(rect[0])
            if camcode == 25053190:
                scaled_cord = [scaled_cord[0] + 292, scaled_cord[1]]
            scaled_coords.append(scaled_cord)

            if class_label.lower() == "potato":
                scaled_WH = scaleWH(rect[1])
                potatoCenterHWarea = [scaled_cord, scaled_WH, poly_area, track_id]
                potato[(x_min, y_min, x_max, y_max)] = potatoCenterHWarea
                potato_sizes.append(poly_area)
                plot_corr_POTATO.append([x_min, y_min, x_max, y_max])

            defect_names.append(class_label)
            # Use uniform green color for all potatoes
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception:
            traceback.print_exc()

    # Process defective potatoes (no tracking)
    boxes_defect = results_defect[0].boxes.xyxy.cpu().numpy()
    classes_defect = results_defect[0].boxes.cls.cpu().numpy()
    names_defect = results_defect[0].names

    for box, cls in zip(boxes_defect, classes_defect):
        try:
            x_min, y_min, x_max, y_max = map(int, box)
            if x_min <= 2 or y_min <= 2 or x_max >= 636 or y_max >= 636:
                continue
            class_label = names_defect[int(cls)]
            rect = get_center_and_size(x_min, y_min, x_max, y_max)

            scale_x = 0.45625  # mm per pixel
            scale_y = 0.49062  # mm per pixel
            width_px, height_px = rect[1]
            width_mm = width_px * scale_x
            height_mm = height_px * scale_y
            potato_size_mm = math.sqrt(width_mm**2 + height_mm**2)
            poly_area = round(potato_size_mm, 2)

            scaled_cord = scalecord(rect[0])
            if camcode == 25053190:
                scaled_cord = [scaled_cord[0] + 292, scaled_cord[1]]
            scaled_coords.append(scaled_cord)

            # Store defective potatoes with no track_id
            potatoCenterHWarea = [scaled_cord, scaleWH(rect[1]), poly_area, None]
            potato[(x_min, y_min, x_max, y_max)] = potatoCenterHWarea
            defect[(x_min, y_min, x_max, y_max)] = [rect[0], class_label, poly_area]
            potato_sizes.append(poly_area)
            plot_corr_POTATO.append([x_min, y_min, x_max, y_max])
            plot_corr_damage.append([x_min, y_min, x_max, y_max])
            defect_areas.append(poly_area)
            defect_names.append(class_label)

            # Visualize defective potatoes
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (int(rect[0][0]), int(rect[0][1])), 10, (10, 20, 255), 2)
            cv2.putText(frame, class_label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception:
            traceback.print_exc()

    print("Potato:", potato)
    print("Defect:", defect)
    resdect = map_defects_to_potatoes(potato, defect)
    # resdect = remove_boundary_potatoes(resdect, img_width=638, img_height=638)
    resdect = merge_good_defective_potatoes(resdect)
    cv2.imwrite("output.png", frame)

    return frame, list(map(float, defect_areas)), list(map(float, potato_sizes)), defect_names, scaled_coords, plot_corr_POTATO, plot_corr_damage, resdect

def string_to_image(base64_string: str) -> np.ndarray:
    """ Convert base64 string to OpenCV image """
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

@app.get("/ServerCheck")
def server_check():
    return {"Server": "OK"}

@app.get("/")
def home():
    return {"Health": "OK"}

@app.post("/def_detection")
async def predictions(request: ImageRequest):
    """ Endpoint for defect detection with image fetched from API """
    try:
        t1 = time.time()
        frame = string_to_image(request.image)
        print("size of image : ",frame.shape)
        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camcode = request.camcode

        # Run detection
        det_image, defect_area, potato_size, defect_name, defect_coords, plot_corr_POTATO, plot_corr_damage, resdect = await detect_defects(frame, camcode)
        # det_image = cv2.cvtColor(det_image, cv2.COLOR_RGB2BGR)
        
        t2 = time.time()
        print(f"Time taken: {(t2 - t1) * 1000:.2f} ms")

        # Create tracked_potatoes list for good potatoes only
        tracked_potatoes = [
            {"id": tid, "box": box, "center": center}
            for tid, box, center in zip(resdect["TrackingID"], resdect["Potato"], resdect["PotatoCenter"])
            if tid != -1
        ]

        response = {
            "defectarea": defect_area,
            "sizeofpotato": potato_size,
            "defectname": defect_name,
            "cord": defect_coords,
            "plot_corr_POTATO": plot_corr_POTATO,
            "plot_corr_damage": plot_corr_damage,
            "tracking_ids": resdect["TrackingID"],
            "tracked_potatoes": tracked_potatoes,
            "resdect": resdect
        }
        print("Response:", response)
        return response

    except Exception as ex:
        traceback.print_exc()
        return {"error": str(ex)}

if __name__ == "__main__":
    uvicorn.run("potato-track:app", host="127.0.0.1", port=5001)