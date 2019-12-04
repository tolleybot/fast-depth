import argparse
import sys
import getopt
import json
from imagezmq import ImageHub
import torch
import torch.nn.parallel
import torch.optim
import cv2
import numpy as np

def run_server(model_file, port, gpu=False):
    """
    """
    if gpu:
        checkpoint = torch.load(model_file, map_location=torch.device('gpu'))
    else:
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    if type(checkpoint) is dict:
        model = checkpoint['model']
    else:
        model = checkpoint

    # setup our server
    image_hub = ImageHub(open_port='tcp://*:{}'.format(port))

    print("Server Started on port {}..\n".format(port))

    while True:
        _, img = image_hub.recv_image()

        img = cv2.resize(img,(224,224))
        img = img.astype('float')
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img).float().to('cpu')

        with torch.no_grad():
            pred = model(img)

        result = pred[0][0].numpy()

        image_hub.send_image('OK',result)


def help():
    print("-m model file")
    print("-t classifier threshold (default: 0.3)")
    print("-u iou threshold (default: 0.5)")


def main(argv):
    try:
        with open('./data/config.json') as f:
            cfg = json.load(f)
    except Exception as e:
        print(str(e))
        return

    model_file = cfg['model']
    port = cfg['port']
    gpu = cfg['gpu']

    try:
        opts, args = getopt.getopt(argv, "hm:p:g",
                                   ["help=", "model=", "port","gpu"])
    except getopt.GetoptError:
        help()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit()
        elif opt in ("-m", "--model"):
            model_file = arg
        elif opt in ("-p", "--port"):
            port = arg
        elif opt in ("-g", "--gpu"):
            gpu = True

    run_server(model_file=model_file, port=port, gpu=gpu)


if __name__ == "__main__":
    main(sys.argv[1:])


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Converted parameters for the model', default='./NYU_ResNet-UpProj.npy')
    parser.add_argument('--port', help='Directory of images to predict', default='5555', type=str)

    args = parser.parse_args()

    print("Starting up server..\n")
    run_server(args.model, args.port)


if __name__ == '__main__':
    main()
