import os
from aiohttp import web
import socketio

# Create Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# Global variable to store broadcaster ID
broadcaster = None

async def index(request):
    try:
        with open('index.html') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except FileNotFoundError:
        return web.Response(text="<html><body><h1>Welcome to WebRTC Server</h1></body></html>", 
                           content_type='text/html')

@sio.event
async def connect(sid, environ):
    print(f'Client connected: {sid}')

@sio.event
async def broadcaster(sid):
    global broadcaster
    broadcaster = sid
    print(f'Broadcaster registered: {broadcaster}')
    await sio.emit('broadcaster-ready', skip_sid=sid)

@sio.event
async def watcher(sid):
    if broadcaster:
        print(f'Watcher connected: {sid}')
        await sio.emit('watcher', sid, room=broadcaster)

@sio.event
async def offer(sid, sdp):
    print('Offer received from broadcaster')
    await sio.emit('offer', sdp, skip_sid=sid)

@sio.event
async def answer(sid, sdp):
    print('Answer received from viewer')
    await sio.emit('answer', sdp, room=broadcaster)

@sio.event
async def candidate(sid, candidate):
    await sio.emit('candidate', candidate, skip_sid=sid)

@sio.event
async def disconnectViewer(sid):
    if broadcaster:
        await sio.emit('viewer-disconnect', sid, room=broadcaster)

@sio.event
async def image(sid, img):
    await sio.emit('image', img)

@sio.event
async def disconnect(sid):
    global broadcaster
    if sid == broadcaster:
        print('Broadcaster disconnected')
        broadcaster = None
        await sio.emit('broadcaster-disconnect', skip_sid=sid)
    else:
        print(f'Client disconnected: {sid}')
        if broadcaster:
            await sio.emit('viewer-disconnect', sid, room=broadcaster)

if os.path.exists('static'):
    app.router.add_static('/static', 'static')

app.router.add_get('/', index)
# Serve the WebRTC client files
app.router.add_static('/', 'test-client-webRTC')

# Start the server
if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 3000))
    print(f'Signaling server running on port {PORT}')
    web.run_app(app, port=PORT)