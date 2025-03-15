import cv2

# Image Load Karo (Ensure karo file path sahi ho)
image = cv2.imread("test_image.jpg")

# Image Show Karo
cv2.imshow("Test Image", image)
cv2.waitKey(0)  # Window ko open rakhne ke liye
cv2.destroyAllWindows()


# Image Load Karo (Ensure karo file path sahi ho)
image = cv2.imread("test_image.jpg")

# Image Show Karo
cv2.imshow("Test Image", image)
cv2.waitKey(0)  # Window ko open rakhne ke liye
cv2.destroyAllWindows()

import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    return lines

def draw_lines(frame, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Live Camera Feed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        edges = preprocess_frame(frame)
        cropped_edges = region_of_interest(edges)
        lines = detect_lines(cropped_edges)
        lane_frame = draw_lines(frame, lines)
        
        cv2.imshow("Lane Assist", lane_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
