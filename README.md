# 🚦 Rakshak 3.0: Smart Traffic Synchronization System

Rakshak 3.0 is an AI-driven traffic management solution that replaces inefficient, static timer-based traffic lights with a dynamic, adaptive system. Utilizing Deep Reinforcement Learning (DQN) and IoT integration, Rakshak optimizes real-time traffic flow and provides instantaneous green-light pre-emption for emergency vehicles.

## 🚀 Key Achievements

- **48.6% Efficiency Gain**: Cleared traffic volume in 368s vs. 716s with traditional static systems
- **Emergency Response**: Instantaneous green-light pre-emption for ambulances
- **Hardware Integration**: Real-time synchronization between SUMO simulation and ESP32 prototype

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|--------------|
| AI/Logic | Python, PyTorch (Deep Q-Learning), NumPy |
| Simulation | SUMO, TraCI API |
| Hardware | ESP32, SH1106 OLED Display, 12-LED Array |
| Communication | Serial (pyserial) @ 115200 Baud |

## 🧠 System Architecture

**1. AI Brain (DQN Model)**
- **State**: Queue lengths, waiting times (4 directions), emergency detection
- **Action**: Optimal lane selection for green signal
- **Reward**: Minimizes cumulative waiting time and vehicle halt duration

**2. IoT Edge (Hardware Prototype)**
- OLED dashboard displaying status and alerts
- LED array representing 4-way intersection states
## DEMO
https://youtu.be/Dow4QunTVik

## 📊 Performance Metrics

| Metric | Static Loop | Rakshak 3.0 |
|--------|------------|------------|
| Total Clearance Time | 716s | 512s |
| Efficiency Improvement | Baseline | +28.49% |
| Emergency Priority | None | Instantaneous |

## 📂 Project Structure

```
├── brain/
│   ├── rakshak_brain.py
│   └── rakshak_final_v2.pth
├── simulation/
│   ├── cfg/
│   └── run_demo.py
└── hardware/
  └── firmware.ino
```

## ⚙️ Installation & Setup

**Hardware Setup:**
1. Connect OLED to ESP32 (SDA: 21, SCL: 22)
2. Connect LEDs to designated GPIOs
3. Upload firmware via Arduino IDE

**Software Setup:**
```bash
pip install torch numpy pyserial traci
```

**Configuration:**
- Update COM port in `run_demo.py`

**Execution:**
```bash

python run_demo.py
```

## 🔮 Future Roadmap

Multi-junction mesh network coordination to create city-wide "Green Corridors" for emergency vehicles.
