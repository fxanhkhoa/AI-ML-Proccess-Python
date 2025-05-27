// DOM elements
const startButton = document.getElementById('startButton');
const hangupButton = document.getElementById('hangupButton');
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const dataChannelInput = document.getElementById('dataChannelInput');
const sendButton = document.getElementById('sendButton');
const receivedMessages = document.getElementById('receivedMessages');

// WebRTC variables
let localStream;
let peerConnection;
let dataChannel;
let roomId;
const receivedCandidates = new Set();
let iceCandidatesBuffer = [];
let mediaStream = new MediaStream();

remoteVideo.onloadedmetadata = () => {
    console.log('Remote video metadata loaded');
    console.log('Video dimensions:', remoteVideo.videoWidth, 'x', remoteVideo.videoHeight);
};

// Check if video is playing
remoteVideo.onplaying = () => {
    console.log('Remote video is now playing!');
};

setInterval(async () => {
    if (!peerConnection) return;
    const stats = await peerConnection.getStats();
    stats.forEach(report => {
        if (report.type === 'inbound-rtp' && report.kind === 'video') {
            console.log('Receiving video data:');
            console.log('- Bytes received:', report.bytesReceived);
            console.log('- Frames decoded:', report.framesDecoded);
            console.log('- Frame rate:', report.framesPerSecond);
        }
    });
}, 2000);

// Socket.io setup
const socket = io('http://localhost:3000');

// ICE servers configuration
const configuration = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
    ]
};

// Socket.io event handlers
socket.on('connect', () => {
    console.log('Connected to signaling server');
});

socket.on('created', room => {
    console.log(`Created room ${room}`);
    roomId = room;
});

socket.on('joined', room => {
    console.log(`Joined room ${room}`);
    roomId = room;
});

socket.on('full', room => {
    console.log(`Room ${room} is full`);
    alert('The room is full. Please try another room.');
});

socket.on('offer', async (description) => {
    try {
        console.log(description, typeof description);
        if (!description) {
            return;
        }
        if (!peerConnection) createPeerConnection();
        // Parse the description if it's a string
        const desc = typeof description === 'string' ? JSON.parse(description) : description;
        if (desc.type === 0) {
            desc.type = 'offer';
        }

        await peerConnection.setRemoteDescription(new RTCSessionDescription(desc));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);

        const cloneAnswer = JSON.parse(JSON.stringify(answer));
        if (cloneAnswer.type === 'answer') {
            // cloneAnswer.type = 2;
        }

        console.log("Answer:", JSON.stringify(cloneAnswer));
        socket.emit('answer', JSON.stringify(cloneAnswer), roomId);
    } catch (error) {
        console.log(error)
    }
});

socket.on('answer', async (description) => {
    const desc = typeof description === 'string' ? JSON.parse(description) : description;
    console.log(desc);
    await peerConnection.setRemoteDescription(new RTCSessionDescription(desc));
    console.log("ANSER SET")
    iceCandidatesBuffer.forEach(candidate => {
        peerConnection.addIceCandidate(candidate);
    });
    iceCandidatesBuffer = [];
});

socket.on('candidate', candidate => {
    try {

        console.log("socket candidate", candidate, typeof candidate)
        const parsed = typeof candidate === 'string' ? JSON.parse(candidate) : candidate;
        console.log(parsed)

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

        if (peerConnection) {
            if (peerConnection.remoteDescription) {
                peerConnection.addIceCandidate(new RTCIceCandidate(parsed))
                    .catch(e => console.error('Error adding received ice candidate', e));
            } else {
                iceCandidatesBuffer.push(parsed);
            }

        }
    } catch (error) {
        console.log(error)
    }
});

// Button event handlers
startButton.addEventListener('click', () => {
    startCall();
});

window.onload = () => {
    roomId = prompt('Enter room name:');
    if (roomId) {
        socket.emit('join', roomId);
    }
};

hangupButton.addEventListener('click', hangup);
sendButton.addEventListener('click', sendData);
dataChannelInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendData();
    }
});

// WebRTC functions
async function startCall() {
    try {
        startButton.disabled = true;
        hangupButton.disabled = false;
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;
        createPeerConnection();

        // Add tracks to the peer connection
        if (peerConnection) {
            localStream.getTracks().forEach(track => {
                peerConnection.addTrack(track, localStream);
            });
            console.log("ADDED TRACK");
        }

        // Create offer if we're the initiator
        createDataChannel();
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        socket.emit('offer', offer, roomId);
    } catch (error) {
        console.error('Error starting call:', error);
    }
}

function createPeerConnection() {
    peerConnection = new RTCPeerConnection({
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
        ],
    });

    // Handle ICE candidates
    peerConnection.onicecandidate = event => {
        if (event.candidate) {
            console.log('ICE candidate:', event.candidate);
            console.log(event.candidate);
            const candidateString = JSON.stringify(event.candidate);
            if (receivedCandidates.has(candidateString)) {
                return;
            }

            receivedCandidates.add(candidateString);

            const candidate = event.candidate;
            const convertedCandidate = {
                "Candidate": candidate.candidate,
                "SdpMid": candidate.sdpMid,
                "SdpMLineIndex": candidate.sdpMLineIndex,
                "Foundation": candidate.foundation,
                "Component": candidate.component,
                "Priority": candidate.priority,
                "Address": candidate.address,
                "Protocol": candidate.protocol,
                "Port": candidate.port,
                "Type": candidate.type,
                "TcpType": candidate.tcpType,
                "RelatedAddress": candidate.relatedAddress,
                "RelatedPort": candidate.relatedPort,
                "UserNameFragment": candidate.userNameFragment
            }

            console.log("SENDING ICE CANDIDATE", candidateString, typeof candidateString);
            socket.emit('candidate', candidateString, roomId);
        }
    };

    // Handle connection state changes
    peerConnection.onconnectionstatechange = event => {
        console.log('Connection state:', peerConnection.connectionState);
    };

    // Handle receiving remote stream
    peerConnection.ontrack = event => {
        if (event.track.kind === 'video') {
            mediaStream.addTrack(event.track);
            localVideo.srcObject = mediaStream;
        }
        console.log('Received ontrack', event);
        if (remoteVideo.srcObject !== event.streams[0]) {
            remoteVideo.srcObject = event.streams[0];
            console.log('Received remote stream');
        }
    };

    // Handle data channel if we're the receiver
    peerConnection.ondatachannel = event => {
        dataChannel = event.channel;
        setupDataChannel();
    };
}

function createDataChannel() {
    dataChannel = peerConnection.createDataChannel('chat');
    setupDataChannel();
}

function setupDataChannel() {
    dataChannel.onopen = () => {
        console.log('Data channel is open');
        dataChannelInput.disabled = false;
        sendButton.disabled = false;
    };

    dataChannel.onclose = () => {
        console.log('Data channel is closed');
        dataChannelInput.disabled = true;
        sendButton.disabled = true;
    };

    dataChannel.onmessage = event => {
        const message = document.createElement('p');
        message.textContent = 'Remote: ' + event.data;
        receivedMessages.appendChild(message);
        receivedMessages.scrollTop = receivedMessages.scrollHeight;
    };
}

function sendData() {
    const message = dataChannelInput.value;
    if (message && dataChannel && dataChannel.readyState === 'open') {
        dataChannel.send(message);

        const messageElement = document.createElement('p');
        messageElement.textContent = 'You: ' + message;
        receivedMessages.appendChild(messageElement);
        receivedMessages.scrollTop = receivedMessages.scrollHeight;

        dataChannelInput.value = '';
    }
}

function hangup() {
    if (dataChannel) {
        dataChannel.close();
        dataChannel = null;
    }

    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localVideo.srcObject = null;
    }

    remoteVideo.srcObject = null;

    socket.emit('leave', roomId);
    roomId = null;

    startButton.disabled = false;
    hangupButton.disabled = true;
    dataChannelInput.disabled = true;
    sendButton.disabled = true;
}