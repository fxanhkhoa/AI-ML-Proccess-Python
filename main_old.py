from aiohttp import web
import socketio
import os
import sys

# Print debug info
print(f"Python version: {sys.version}")
print(f"socketio version: {socketio}")

# Create Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
app = web.Application()
sio.attach(app)

# Store active rooms
rooms = {}

# Serve static files
# Serve static files
async def index(request):
    try:
        with open('index.html') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except FileNotFoundError:
        return web.Response(text="<html><body><h1>Welcome to WebRTC Server</h1></body></html>", 
                           content_type='text/html')

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f'Client connected: {sid}')
    await sio.emit('my_response', {'data': 'Connected to signaling server'}, room=sid)

@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')
    # Clean up any rooms the user was in
    for room_id, participants in list(rooms.items()):
        if sid in participants:
            participants.remove(sid)
            if not participants:
                del rooms[room_id]


@sio.event
async def join(sid, room_id):
    print(f'Client {sid} is joining room: {room_id}')
    
    if room_id not in rooms:
        # Create new room
        rooms[room_id] = {sid}
        await sio.emit('created', room_id, room=sid)
    elif len(rooms[room_id]) < 2:
        # Join existing room
        rooms[room_id].add(sid)
        await sio.enter_room(sid, room_id)
        await sio.emit('joined', room_id, room=sid)
    else:
        # Room is full
        await sio.emit('full', room_id, room=sid)

@sio.event
async def offer(sid, offer, room_id):
    print(f'Received offer from {sid} for room {room_id}')
    # Forward the offer to the other peer in the room
    for participant in rooms.get(room_id, set()):
        if participant != sid:
            await sio.emit('offer', offer, room=participant)

@sio.event
async def answer(sid, answer, room_id):
    print(f'Received answer from {sid} for room {room_id}')
    # Forward the answer to the other peer in the room
    for participant in rooms.get(room_id, set()):
        if participant != sid:
            await sio.emit('answer', answer, room=participant)

@sio.event
async def candidate(sid, candidate, room_id):
    print(f'Received ICE candidate from {sid}')
    # Forward the ICE candidate to the other peer in the room
    for participant in rooms.get(room_id, set()):
        if participant != sid:
            await sio.emit('candidate', candidate)

@sio.event
async def leave(sid, room_id):
    print(f'Client {sid} is leaving room: {room_id}')
    if room_id in rooms and sid in rooms[room_id]:
        rooms[room_id].remove(sid)
        await sio.leave_room(sid, room_id)
        if not rooms[room_id]:
            del rooms[room_id]
        else:
            # Notify other participants that a peer has left
            for participant in rooms[room_id]:
                await sio.emit('peer_left', room=participant)

# Set up static routes and the index page
# Check if directories exist before adding routes
if os.path.exists('static'):
    app.router.add_static('/static', 'static')

app.router.add_get('/', index)
# Serve the WebRTC client files
app.router.add_static('/', 'test-client-webRTC')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))  # Match the port in the client JS
    web.run_app(app, port=port)
    print(f'Server running on http://localhost:{port}')