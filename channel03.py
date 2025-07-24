# simple_pub.py
import time
import zmq

from analytics_channel import channel_analytics
from utils import image_to_base64
host = "127.0.0.1"
port = "5001"
# Creates a socket instance
context = zmq.Context()
socket = context.socket(zmq.PUB)
# Binds the socket to a predefined port on localhost
socket.bind("tcp://{}:{}".format(host, port))
# Sends a string message

CHANNEL_NAME = "ch03"
SHOW_FRAME = True
START_FRAME = 0
FPS = 30
CASHIER_RECOGNITION = True
DISPLAY_REGION = True

# while True:
#     socket.send_json({
#         "test": "test"
#     })
#     sleep(1/30)

while True:
    for output in channel_analytics(
        channel_name=CHANNEL_NAME,
        show_frame=SHOW_FRAME,
        start_frame=START_FRAME,
        cashier_recognition=CASHIER_RECOGNITION,
        display_region=DISPLAY_REGION
    ):
        frame = output["frame"]

        if SHOW_FRAME:
            frame = image_to_base64(frame)

        socket.send_json(
            {"frame": frame, "metadata": output["metadata"]}
        )

        # sleep(1/FPS)
