# Potato Defect Detection System Using YOLOv8

## Description
This project implements a real-time defect detection system for potatoes using YOLOv8 models, designed for agricultural quality control. It leverages two YOLOv8 models—one for detecting good potatoes and another for identifying defects—running on GPU (if available) with PyTorch for optimized performance. The system processes base64-encoded images received via a FastAPI endpoint, resizes and converts them for detection, and uses the Supervision library’s ByteTrack for tracking good potatoes. Defects are mapped to potatoes based on spatial proximity, and their areas are calculated in mm² using pixel-to-mm conversion. The system filters out potatoes touching image boundaries and merges good and defective detections to ensure accurate classification. Results include bounding boxes, defect areas, potato sizes, tracking IDs, and scaled coordinates for downstream applications, all returned via a REST API. The project is optimized for scalability and reliability, with comprehensive error handling and logging.

## Features
- **Object Detection**: Uses YOLOv8 models to detect good potatoes and defects with high accuracy.
- **Real-Time Tracking**: Implements ByteTrack to track good potatoes across frames.
- **Defect Mapping**: Maps defects to potatoes based on center inclusion and proximity.
- **Area Calculation**: Computes potato and defect areas in mm² using pixel-to-mm scaling.
- **API Integration**: Provides a FastAPI endpoint for processing base64-encoded images.
- **GPU Optimization**: Leverages PyTorch for GPU-accelerated inference.

## Tech Stack
- **Languages**: Python
- **Libraries**: YOLOv8, FastAPI, PyTorch, OpenCV, Supervision, NumPy, PIL
- **Tools**: REST API, GPU Acceleration, Object Tracking

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rkarahul/Potato-Defect-Detection-System.git
   cd Potato-Defect-Detection-System
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Place the YOLOv8 model weights (`best_defect.pt`, `best_good.pt`) in the `models/` directory.

## Usage
1. Run the FastAPI server:
   ```bash
   python src/main.py
   ```
2. Send a POST request to `http://127.0.0.1:5001/def_detection` with a JSON payload containing a base64-encoded image and camera code:
   ```json
   {
       "image": "base64_string_here",
       "camcode": 25053190
   }
   ```
3. Receive the detection results, including defect areas, potato sizes, and tracking IDs.

## File Structure and Functionality
The project is organized into modular files for clarity and maintainability. Below is a breakdown of the key files and their roles:

- **`src/main.py`**: The entry point of the application. This script sets up the FastAPI app, defines endpoints (`/ServerCheck`, `/`, `/def_detection`), and handles incoming requests. The `/def_detection` endpoint processes base64-encoded images, invokes the detection pipeline, and returns the results, including tracking IDs and defect mappings.

- **`src/detection.py`**: Contains the core detection logic using YOLOv8 models. Key functions include:
  - `detect_defects()`: Runs inference on both YOLO models to detect good potatoes and defects, applies tracking with ByteTrack, and visualizes results on the image.
  - Processes detection outputs, calculates scaled coordinates, and prepares data for mapping.

- **`src/tracking.py`**: Manages tracking and mapping of potatoes and defects. Key functions include:
  - `map_defects_to_potatoes()`: Maps defects to potatoes based on center inclusion within bounding boxes.
  - `remove_boundary_potatoes()`: Filters out potatoes touching image boundaries to improve accuracy.
  - `merge_good_defective_potatoes()`: Merges good and defective potatoes based on proximity to handle overlapping detections.
  - `get_center_and_size()`: Calculates the center and size of bounding boxes for mapping and visualization.

- **`src/utils.py`**: Provides utility functions for image processing and calculations. Key functions include:
  - `image_to_base64()`: Converts an OpenCV image to a base64 string.
  - `string_to_image()`: Converts a base64 string to an OpenCV image for processing.
  - `calculate_polygon_area()`: Computes the area of a polygon in mm² using pixel-to-mm conversion.
  - `scalecord()`: Scales coordinates to a 300x300 field for downstream applications.
  - `scaleWH()`: Scales width and height of bounding boxes to a 300x300 field.

- **`models/best_defect.pt`**: YOLOv8 model weights for detecting defects on potatoes.
- **`models/best_good.pt`**: YOLOv8 model weights for detecting good potatoes.
- **`data/sample_image.jpg`**: A sample input image for testing the detection pipeline (if included).
- **`tests/test_detection.py`**: Includes unit tests for detection and tracking functions (to be implemented).
- **`requirements.txt`**: Lists all Python dependencies required to run the project.

This modular structure ensures each component is isolated, making the codebase easy to maintain and extend.

## Example Output
```json
{
    "defectarea": [12.34, 15.67],
    "sizeofpotato": [50.12, 48.90],
    "defectname": ["blemish", "crack"],
    "cord": [[145, 157], [200, 210]],
    "plot_corr_POTATO": [[100, 120, 200, 220], [300, 320, 400, 420]],
    "plot_corr_damage": [[150, 160, 180, 190]],
    "tracking_ids": [1, 2],
    "tracked_potatoes": [
        {"id": 1, "box": [100, 120, 200, 220], "center": [150, 170]},
        {"id": 2, "box": [300, 320, 400, 420], "center": [350, 370]}
    ],
    "resdect": {
        "Potato": [[100, 120, 200, 220], [300, 320, 400, 420]],
        "PotatoArea": [50.12, 48.90],
        "Potato_WH": [[100, 100], [100, 100]],
        "PotatoCenter": [[150, 170], [350, 370]],
        "TrackingID": [1, 2],
        "defectname": [["blemish"], ["crack"]],
        "DefectArea": [[12.34], [15.67]],
        "DefectCord": [[[150, 160, 180, 190]], [[200, 210, 230, 240]]]
    }
}
```

## Project Structure
- `src/`: Core source code for detection, tracking, and API handling.
- `models/`: YOLOv8 model weights.
- `data/`: Sample input data.
- `tests/`: Unit tests for key functions.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## Contact
For questions, reach out to [your.email@example.com](mailto:your.email@example.com).
