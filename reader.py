# simple_sub.py
import zmq
host = "127.0.0.1"
port = "5001"
# Creates a socket instance
context = zmq.Context()
socket = context.socket(zmq.SUB)
# Connects to a bound socket
socket.connect("tcp://{}:{}".format(host, port))
# Subscribes to all topics
socket.subscribe("")
# Receives a string format message
while True:
    message = socket.recv_json()
    print(message)