import socket
import time
from imagezmq import ImageSender
import cv2
import argparse
import json
import numpy as np

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', help='port to use', default='5555',type=str)
    parser.add_argument('--host', help='server IP or name', default='localhost', required=False)
    args = parser.parse_args()

    sender = ImageSender(connect_to='tcp://'+args.host+':' + args.port)

    cap = cv2.VideoCapture(0)

    while True:  # send images as stream until Ctrl-C

        ret, frame = cap.read()

        if not ret:
            print("error with video")
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        msg, results = sender.send_image2("image", frame)

        c = results.shape[1] * 2
        r = results.shape[0] * 2

        img = cv2.resize(results, dsize=(c, r), interpolation=cv2.INTER_CUBIC)
        minv = np.min(img)
        maxv = np.max(img)

        img *= 255 / maxv
        img = img.astype(np.uint8)

        #print(img)

        cv2.imshow('depth', img)
        cv2.waitKey(1)





if __name__ == '__main__':
    main()
