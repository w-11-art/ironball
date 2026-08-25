import cv2

for i in range(5):

    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"发现摄像头：{i}")

        ret, frame = cap.read()

        if ret:
            print(
                f"摄像头 {i} 可以正常读取画面"
            )

            cv2.imshow(
                f"Camera {i}",
                frame
            )

            cv2.waitKey(2000)

        cap.release()

cv2.destroyAllWindows()