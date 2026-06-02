import cv2
import numpy as np

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)

        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))

    lines_out = []

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lines_out.append(make_coordinates(img, left_avg))

    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lines_out.append(make_coordinates(img, right_avg))

    return lines_out


def display_lines(img, lines):
    line_image = np.zeros_like(img)

    if lines:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    return line_image

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    median = np.median(blur)
    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))

    canny = cv2.Canny(blur, lower, upper)

    cropped = region_of_interest(canny)

    lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 50,
                            minLineLength=100, maxLineGap=50)

    averaged_lines = average_slope_intercept(frame, lines)

    if averaged_lines == []:
        return frame

    line_image = display_lines(frame, averaged_lines)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return combo

