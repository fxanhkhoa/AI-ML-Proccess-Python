const imageElement = document.getElementById('imageElement');

const socket = io('http://localhost:3000');

socket.on('connect', () => {
    console.log("connected")
});

socket.on('disconnect', () => {
    console.log("disconnected")
});

socket.on('image_broadcast', (imageBase64) => {
    console.log('Received image');
    imageElement.src = `data:image/jpeg;base64,${imageBase64}`;
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    socket.emit('disconnectViewer');
    socket.disconnect();
});