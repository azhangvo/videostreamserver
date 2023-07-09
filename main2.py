import socketio
import time
from aiohttp import web

sio = socketio.Server()
app = web.Application

clients = []


@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)
    clients.append(sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)
    clients.remove(sid)


web.

while True:
    print("Connected clients:")
    print(clients)

    time.sleep(2)
