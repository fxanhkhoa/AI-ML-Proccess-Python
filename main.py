import socketio
import aiohttp
from aiohttp import web
import base64
import os
from src.image_storage import set_image
from src.process import process_start
from src.type.engine import EngineUpdate

# Create a Socket.IO server
sio = socketio.AsyncServer(cors_allowed_origins='*')
app = web.Application()
sio.attach(app)
client_id = ''

async def index(request):
    try:
        with open('index.html') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except FileNotFoundError:
        return web.Response(text="<html><body><h1>Welcome to WebRTC Server</h1></body></html>", 
                           content_type='text/html')

# Event handler for disconnections
@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# Event handler for receiving and broadcasting an image
@sio.event
async def broadcast_image(sid, data):
    # Assuming data contains base64 encoded image
    # You could also encode an image here if needed:
    # with open('image.jpg', 'rb') as img_file:
    #     base64_img = base64.b64encode(img_file.read()).decode('utf-8')
    
    print(f"Broadcasting image from {sid}")
    set_image(data)
    # Broadcast the image to all connected clients (except sender)
    await sio.emit('image_broadcast', data, skip_sid=sid)

if os.path.exists('static'):
    app.router.add_static('/static', 'static')

app.router.add_get('/', index)
# Serve the WebRTC client files
app.router.add_static('/', 'test-client-webRTC')

# Set up routes
app.router.add_get('/', index)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))

    @sio.on('connect')
    async def handle_connect(sid, environ):
        global client_id
        # Start a background task to process images
        client_id = sid
        sio.start_background_task(process_images)

    async def process_images():
        global client_id
        while True:
            await process_start(sio, client_id)
            # Add a small delay to prevent CPU overuse
            await sio.sleep(2)

    print(f'Server running on http://localhost:{port}')
    web.run_app(app, port=port)