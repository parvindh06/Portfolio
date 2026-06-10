import cv2
import numpy as np
import requests

STREAM_URL = "http://192.168.4.1:81/stream"
THERMAL_URL = "http://192.168.4.1/thermal"

GRID_ROWS = 4
GRID_COLS = 4

GREEN_LOW = (35, 40, 40)
GREEN_HIGH = (85, 255, 255)

MAX_VEG_DENSITY = 0.25
MIN_SOIL_TEMP = 18.0
MAX_SOIL_TEMP = 30.0


def vegetation_density(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    return np.count_nonzero(mask) / mask.size


def plant_shapes(cell, min_area=150):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            shapes.append(c)
    return shapes


def read_thermal_grid():
    try:
        response = requests.get(THERMAL_URL, timeout=0.3)
        values = [float(v) for v in response.text.strip().split(",")]
        return np.array(values).reshape(8, 8)
    except Exception:
        return np.full((8, 8), 22.0)


def soil_temp_for_cell(grid, row, col):
    r0 = int(row * 8 / GRID_ROWS)
    r1 = int((row + 1) * 8 / GRID_ROWS)
    c0 = int(col * 8 / GRID_COLS)
    c1 = int((col + 1) * 8 / GRID_COLS)
    return float(grid[r0:r1, c0:c1].mean())


def is_plantable(veg, temp):
    return veg < MAX_VEG_DENSITY and MIN_SOIL_TEMP <= temp <= MAX_SOIL_TEMP


def open_stream():
    cap = cv2.VideoCapture(STREAM_URL)
    if cap.isOpened():
        print("Connected to ESP32-CAM.")
        return cap
    print("ESP32-CAM not found. Using webcam.")
    return cv2.VideoCapture(0)


def main():
    cap = open_stream()
    if not cap.isOpened():
        print("No video source. Exiting.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame. Stopping.")
            break

        height, width = frame.shape[:2]
        cell_h = height // GRID_ROWS
        cell_w = width // GRID_COLS
        thermal = read_thermal_grid()
        plant_count = 0

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                y0, y1 = row * cell_h, (row + 1) * cell_h
                x0, x1 = col * cell_w, (col + 1) * cell_w
                cell = frame[y0:y1, x0:x1]

                veg = vegetation_density(cell)
                temp = soil_temp_for_cell(thermal, row, col)

                if is_plantable(veg, temp):
                    plant_count += 1
                    color = (0, 255, 0)
                    label = "PLANT"
                else:
                    color = (0, 0, 255)
                    label = "SKIP"

                for shape in plant_shapes(cell):
                    cv2.drawContours(frame, [shape + [x0, y0]], -1, (255, 255, 0), 1)

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, label,
                            (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"Green {veg:.0%}  Soil {temp:.0f}C",
                            (x0 + 6, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 255), 1)

        cv2.putText(frame, f"Plantable zones: {plant_count} of {GRID_ROWS * GRID_COLS}",
                    (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Seedling Drone  |  Field Map  |  ESC to quit", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()