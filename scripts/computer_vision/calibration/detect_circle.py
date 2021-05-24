import cv2
import numpy as np


def detect_circle(frame, delta):
    w, h, d = np.shape(frame)
    x, y = 0, 0

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT,
                               4, 600, minRadius=90, maxRadius=120)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.circle(output, (int(h/2), int(w/2)), delta, (0, 0, 255), 4)
            cv2.rectangle(output, (int(w/2) - 5, int(h/2) - 5), (int(w/2) + 5, int(h/2) + 5), (0, 0, 255), -1)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.line(output, (0, int(w/2)), (h, int(w/2)), (0, 255, 255), 2)
            cv2.line(output, (int(h/2), 0), (int(h/2), w), (0, 255, 255), 2)
        # show the output image

    return output, [x, y]


def main():

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            output, points = detect_circle(frame)
            cv2.imshow("output", output)
            print(points)
        else:
            cv2.imshow('test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()