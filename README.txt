Soul Digital Twin & AI Personality System
========================================

Overview
--------
This project is an interactive digital twin and AI personality simulator for robotics and electronics. It allows users to visually assemble a robot (microcontroller + modules) via drag-and-drop, and simulates how an AI's internal traits and personality evolve in response to its hardware configuration. The system features real-time trait evolution, AI self-reflection, and a 3D visualization of the robot's body.

Key Features
------------
- **Drag-and-drop prototyping:** Add, move, and connect microcontrollers and modules (sensors, actuators, etc.) in a visual canvas.
- **Live trait evolution:** The AI's core traits (perception, agency, memory, etc.) dynamically change based on the hardware configuration and internal simulation.
- **AI self-discovery:** When new modules are added, the AI uses Gemini (LLM) to "learn" about them and updates its self-description/personality.
- **Digital twin simulation:** The backend can simulate the robot's operation and return summaries of its behavior.
- **3D visualization:** A Three.js-powered 3D view shows the robot's body and modules in space.
- **Randomize hardware:** A button to continuously add/remove modules, letting you watch the AI's traits and personality adapt in real time.

Architecture
------------
- **Frontend:** HTML/JS (with Plotly for trait plots, Three.js for 3D, and vanilla JS for UI logic)
- **Backend:** FastAPI (Python) with endpoints for trait updates, hardware analysis, AI code suggestion, and digital twin simulation
- **AI/LLM:** Google Gemini API for module descriptions and self-reflection

How It Works
------------
1. **User assembles robot:** Drag modules and microcontroller(s) onto the canvas, connect them as desired.
2. **Hardware-to-trait mapping:** Each module type influences one or more AI traits (e.g., camera → perception, motor → agency).
3. **Trait evolution:** The AI's trait values are a blend of hardware-driven baselines and internal simulation (random walk, learning, etc.).
4. **Self-discovery:** When new modules are added, the backend uses Gemini to fetch a description and generate a first-person personality summary.
5. **Digital twin simulation:** The backend can simulate the robot's operation, returning summaries of its behavior and trait changes.
6. **Visualization:** The UI updates trait bars, personality summary, and (optionally) a 3D view in real time.

Theories and Math Used
----------------------
- **Trait Vector Model:**
  - The AI's "soul" is represented as a vector of trait values (e.g., perception, agency, memory, etc.), each in [0, 1].
  - Hardware modules provide a baseline for each trait (via a mapping), and the trait vector evolves over time.
- **Cosine Similarity:**
  - Used to compare the current trait vector to theoretical models of consciousness (IIT, GWT, etc.).
- **Random Walks:**
  - Trait values fluctuate over time using a bounded random walk, simulating internal development and adaptation.
- **Correlation Analysis:**
  - The system can analyze correlations between traits over time to detect emergent patterns.
- **LLM-based Reasoning:**
  - Google Gemini is used for module descriptions, self-reflection, and theory updates, providing natural language insights.
- **Digital Twin Simulation:**
  - The backend can simulate the robot's operation as a graph of modules and connections, optionally using ML or rule-based logic for more advanced behavior.

Personality and Self-Reflection
-------------------------------
- The AI generates a first-person summary of its personality and capabilities based on its current hardware and trait values.
- When new modules are added, it "learns" about them and updates its self-concept, simulating a form of self-discovery.

Extending the System
--------------------
- Add more module types and trait mappings for richer behavior.
- Integrate a physics engine (Cannon.js, Rapier) for 3D/physics-based digital twin simulation.
- Expand the digital twin logic to simulate real sensor/actuator data and closed-loop control.
- Use reinforcement learning or more advanced AI for adaptive behavior.

Contact & Credits
-----------------
- Built with FastAPI, Plotly, Three.js, and Google Gemini.
- For questions or contributions, see the project repository or contact the author. 