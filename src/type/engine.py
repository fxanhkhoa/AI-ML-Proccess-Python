import json
import time

from socketio import AsyncServer

class EngineUpdate:
    def __init__(self, acceleration: float = 0.0, steerAngle: float = 0.0, brake: float = 0.0, acceleration_interval = 3):
        self.acceleration = acceleration
        self.steerAngle = steerAngle
        self.brake = brake
        self.acceleration_interval = acceleration_interval
        self.start_time = time.time()

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
    
    async def emit_to_simulation(self, sio: AsyncServer, client_id: str):
        current_time = time.time() - self.start_time
        is_acceleration_on = int(current_time / self.acceleration_interval) % 2 == 0

        current_state = {
            'acceleration': self.acceleration if is_acceleration_on else 0.0,
            'steerAngle': self.steerAngle,
            'brake': self.brake
        }

        print('emitting', client_id, current_state)
        
        await sio.emit(event='engine_update', data=json.dumps(current_state))
        # sio.send(event='engine_update', data=json.dumps(current_state), to=client_id)