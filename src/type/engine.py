import json

from socketio import AsyncServer

class EngineUpdate:
    def __init__(self, acceleration: float = 0.0, steerAngle: float = 0.0, brake: float = 0.0):
        self.acceleration = acceleration
        self.steerAngle = steerAngle
        self.brake = brake

    @classmethod
    def from_json_string(cls, json_string: str) -> 'EngineUpdate':
        """Create an EngineUpdate instance from a JSON string."""
        data = json.loads(json_string)
        return cls(
            acceleration=float(data.get('acceleration', 0.0)),
            steerAngle=float(data.get('steerAngle', 0.0)),
            brake=float(data.get('brake', 0.0))
        )

    def to_json_string(self) -> str:
        """Convert the EngineUpdate instance to a JSON string."""
        return json.dumps({
            'acceleration': self.acceleration,
            'steerAngle': self.steerAngle,
            'brake': self.brake
        })
    
    def emit_to_simulation(self, sio: AsyncServer, client_id: str):
        sio.emit(event='engine_update', data=self.to_json_string(), to=client_id)