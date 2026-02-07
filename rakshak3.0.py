import traci
import time
import numpy as np
import os
import sys
import serial  # For ESP32 Communication
from rakshak_brain import RakshakAgent

# ==========================================
#   🌟 RAKSHAK 3.0: THE ULTIMATE DEMO
# ==========================================

# --- SERIAL SETUP (Hardware Connection) ---
COM_PORT = 'COM19'  # Your Laptop's Port
BAUD_RATE = 115200

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"📡 Hardware Connected: {COM_PORT}")
    time.sleep(2) # Wait for ESP32 to reboot
except Exception as e:
    print(f"⚠️ Hardware Error: Could not connect to {COM_PORT}. Check cable/drivers.")
    ser = None

SUMO_CMD = ["sumo-gui", "-c", "sumo_intersection/cfg/data2_simulation.sumocfg"]
STATE_SIZE = 8   
ACTION_SIZE = 4  
MIN_GREEN_TIME = 6 
YELLOW_TIME = 4   

LANES = {
    "N": ["A0_0", "A0_1", "A0_2"],
    "E": ["D0_0", "D0_1", "D0_2"],
    "S": ["C0_0", "C0_1", "C0_2"],
    "W": ["B0_0", "B0_1", "B0_2"]
}

PHASE_GREEN = {0: 0, 1: 3, 2: 6, 3: 9}
PHASE_YELLOW = {0: 1, 1: 4, 2: 7, 3: 10}
NAME_MAP = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}

# --- LOAD BRAIN ---
print("🧠 Initializing Rakshak 3.0 AI...")
agent = RakshakAgent(STATE_SIZE, ACTION_SIZE)
model_path = "rakshak_final_brain_v2.pth"

if os.path.exists(model_path):
    agent.load(model_path)
    print("✅ SUCCESS: Brain Loaded & Ready.")
else:
    print("❌ ERROR: Brain file not found. Train first!")
    sys.exit()

def send_to_hardware(command):
    """Sends action code to ESP32"""
    if ser and ser.is_open:
        ser.write(f"{command}\n".encode())

def get_state():
    queues = []; ambs = []
    for direction in ["N", "E", "S", "W"]:
        q = 0; a = 0
        for lane in LANES[direction]:
            try:
                q += traci.lane.getLastStepHaltingNumber(lane)
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    try:
                        if "amb" in v or traci.vehicle.getVehicleClass(v) == "emergency": 
                            a = 1
                    except: pass
            except: pass
        queues.append(min(q / 20.0, 1.0)) 
        ambs.append(a)
    return np.array(queues + ambs)

# --- START SIMULATION ---
traci.start(SUMO_CMD)
step = 0
last_switch_step = 0
current_action = 0 

# Initial Hardware Sync
send_to_hardware(0) 
traci.trafficlight.setPhase("J", PHASE_GREEN[0])

print("\n--- 🚦 RAKSHAK 3.0 LIVE DASHBOARD ---")
print(f"{'STEP':<8} | {'EVENT TYPE':<15} | {'ACTION / STATUS':<30}")
print("-" * 60)

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    state = get_state()
    
    # --- 1. AMBULANCE CHECK (GOD MODE) ---
    amb_active = False
    for i in range(4):
        if state[4+i] > 0: 
            if current_action != i:
                print(f"{step:<8} | 🚨 AMBULANCE    | ⚡ INSTANT SWITCH -> {NAME_MAP[i]}")
                traci.trafficlight.setPhase("J", PHASE_GREEN[i])
                send_to_hardware(i) # Sync Hardware
                current_action = i
                last_switch_step = step
                amb_active = True
            else:
                last_switch_step = step 
                amb_active = True
            break
            
    # --- 2. AI DECISION (NORMAL TRAFFIC) ---
    if not amb_active and (step - last_switch_step) > MIN_GREEN_TIME:
        agent.epsilon = 0.0 
        action = agent.act(state)
        
        if action != current_action:
            # YELLOW TRANSITION
            traci.trafficlight.setPhase("J", PHASE_YELLOW[current_action])
            send_to_hardware(current_action + 4) # 4-7 are Yellow commands
            print(f"{step:<8} | 🟡 YELLOW LIGHT   | Leaving {NAME_MAP[current_action]}...")
            
            for _ in range(YELLOW_TIME):
                traci.simulationStep()
                step += 1
            
            # GREEN TRANSITION
            traci.trafficlight.setPhase("J", PHASE_GREEN[action])
            send_to_hardware(action) # 0-3 are Green commands
            print(f"{step:<8} | 🟢 GREEN LIGHT    | {NAME_MAP[action]}")
            
            current_action = action
            last_switch_step = step

    step += 1
    time.sleep(0.02) 

traci.close()
if ser: ser.close()
print("🛑 DEMO COMPLETE")