const videoElement = document.getElementById('videoElement');
const statusBox = document.getElementById('statusBox');
let peerConnection;

// Connect to signaling server
const socket = io('http://localhost:3000');

socket.on('connect', () => {
    setStatus('Connected to signaling server', true);

    // Tell the server we're a viewer
    socket.emit('watcher');
});

socket.on('disconnect', () => {
    setStatus('Disconnected from signaling server', false);
    closePeerConnection();
});

socket.on('offer', (sdp) => {
    setStatus('Received offer from broadcaster', true);
    handleOffer(sdp);
});

socket.on('candidate', candidate => {
    console.log(candidate)
    if (!candidate) return;
    const lowerCaseCandidate =
    {
        "candidate": candidate.Candidate,
        "sdpMid": candidate.SdpMid,
        "sdpMLineIndex": candidate.SdpMLineIndex,
        "foundation": candidate.Foundation,
        "component": candidate.Component,
        "priority": candidate.Priority,
        "address": candidate.Address,
        "protocol": candidate.Protocol,
        "port": candidate.Port,
        "type": candidate.Type,
        "tcpType": candidate.TcpType,
        "relatedAddress": candidate.RelatedAddress,
        "relatedPort": candidate.RelatedPort,
        "userNameFragment": candidate.UserNameFragment
    }
    if (!peerConnection) return;
    try {
        const iceCandidate = new RTCIceCandidate(lowerCaseCandidate);

        peerConnection.addIceCandidate(iceCandidate)
            .catch(error => console.error('Error adding ICE candidate:', error));
    } catch (error) {
        console.error('Error handling ICE candidate:', error);
    }
});

// Handle broadcaster disconnection
socket.on('broadcaster-disconnect', () => {
    setStatus('Broadcaster disconnected', false);
    closePeerConnection();
});

async function handleOffer(sdp) {
    try {
        if (peerConnection) {
            closePeerConnection();
        }

        createPeerConnection();

        await peerConnection.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp }));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);

        socket.emit('answer', answer.sdp);
        setStatus('Sent answer to broadcaster', true);
    } catch (error) {
        console.error('Error handling offer:', error);
        setStatus('Error connecting to stream: ' + error.message, false);
    }
}

function createPeerConnection() {
    const config = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' }
        ]
    };

    peerConnection = new RTCPeerConnection(config);

    peerConnection.ontrack = (event) => {
        console.log("Track received:", event);

        if (event.track.kind === 'video') {
            console.log("Video track received");

            // Check if the stream has tracks
            if (event.streams && event.streams[0]) {
                console.log("Stream has tracks:", event.streams[0].getTracks().length);

                const videoElement = document.getElementById('videoElement');
                videoElement.srcObject = event.streams[0];

                // Add event listeners to debug video element
                videoElement.onloadedmetadata = () => {
                    console.log("Video metadata loaded",
                        videoElement.videoWidth + "x" + videoElement.videoHeight);
                };

                videoElement.onplay = () => {
                    console.log("Video started playing");
                };

                videoElement.onerror = (e) => {
                    console.error("Video error:", e);
                };

                // Force play (might be needed in some browsers)
                videoElement.play().catch(e => console.error("Error playing video:", e));
            } else {
                console.warn("Received track without a stream");
            }
        }
    };

    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            socket.emit('candidate', JSON.stringify(event.candidate));
        }
    };

    peerConnection.oniceconnectionstatechange = () => {
        console.log('ICE connection state:', peerConnection.iceConnectionState);
        if (peerConnection.iceConnectionState === 'disconnected' ||
            peerConnection.iceConnectionState === 'failed' ||
            peerConnection.iceConnectionState === 'closed') {
            setStatus('Connection lost', false);
        }
        else if (peerConnection.iceConnectionState === 'connected') {
            setStatus('Connected to stream', true);
        }
    };
}

function closePeerConnection() {
    if (peerConnection) {
        peerConnection.ontrack = null;
        peerConnection.onicecandidate = null;
        peerConnection.oniceconnectionstatechange = null;
        peerConnection.close();
        peerConnection = null;
    }

    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
}

function setStatus(message, connected) {
    statusBox.textContent = message;
    statusBox.className = 'status ' + (connected ? 'connected' : 'disconnected');
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    socket.emit('disconnectViewer');
    closePeerConnection();
    socket.disconnect();
});