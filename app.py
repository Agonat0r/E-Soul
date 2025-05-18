import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template
from flask_socketio import SocketIO
from soul_simulation import SoulSimulation
import json
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)
simulation = SoulSimulation()

def datetime_handler(x):
    if isinstance(x, datetime):
        return x.isoformat()
    raise TypeError(f"Object of type {type(x)} is not JSON serializable")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    # Send initial state
    state = simulation.get_simulation_state()
    socketio.emit('simulation_update', {
        'traits': [{
            'name': t.name,
            'value': t.value,
            'description': t.description,
            'category': t.category,
            'created_at': t.created_at.isoformat()
        } for t in state['traits']],
        'theories': [{
            'name': t.name,
            'description': t.description,
            'required_traits': t.required_traits,
            'confidence': t.confidence,
            'last_updated': t.last_updated.isoformat()
        } for t in state['theories']]
    })

def run_simulation():
    while True:
        simulation.run_simulation_step()
        state = simulation.get_simulation_state()
        socketio.emit('simulation_update', {
            'traits': [{
                'name': t.name,
                'value': t.value,
                'description': t.description,
                'category': t.category,
                'created_at': t.created_at.isoformat()
            } for t in state['traits']],
            'theories': [{
                'name': t.name,
                'description': t.description,
                'required_traits': t.required_traits,
                'confidence': t.confidence,
                'last_updated': t.last_updated.isoformat()
            } for t in state['theories']]
        })
        socketio.sleep(2)  # Update every 2 seconds

if __name__ == '__main__':
    socketio.start_background_task(run_simulation)
    socketio.run(app, debug=True) 