import traci
import numpy as np
import os
import random
import sys
import time
from rakshak_brain import RakshakAgent

# ==========================================
#   🎯 HARDCODED CONFIGURATION (NO GUESSING)
# ==========================================
STATE_SIZE = 8   
ACTION_SIZE = 4  # N, E, S, W
BATCH_SIZE = 64
EPISODES = 30
MAX_STEPS = 3600

# Explicit Lanes (Matches your network)
LANES = {
    "N": ["A0_0", "A0_1", "A0_2"],
    "E": ["D0_0", "D0_1", "D0_2"],
    "S": ["C0_0", "C0_1", "C0_2"],
    "W": ["B0_0", "B0_1", "B0_2"]
}

# Phase Mapping
PHASE_MAP = {
    0: 0,  # North Green
    1: 3,  # East Green
    2: 6,  # South Green
    3: 9   # West Green
}

# --- INITIALIZE AI ---
agent = RakshakAgent(STATE_SIZE, ACTION_SIZE)
agent.epsilon = 1.0 
agent.epsilon_decay = 0.90 

if not os.path.exists("models"): os.makedirs("models")

def inject_ambulance(step):
    """Spawns ambulance securely by checking available types"""
    if random.random() < 0.005: 
        try:
            # 1. Get a valid vehicle type that actually exists
            valid_types = traci.vehicletype.getIDList()
            if not valid_types: return # No types defined? Skip.
            
            # Use the first available type as a base (e.g., 'DEFAULT_VEHTYPE')
            v_type = valid_types[0] 
            
            # 2. Pick a random route
            routes = traci.route.getIDList()
            if not routes: return # No routes? Skip.
            r = random.choice(routes)
            
            # 3. Spawn
            vid = f"amb_{step}"
            traci.vehicle.add(vid, routeID=r, typeID=v_type)
            
            # 4. Make it look like an ambulance
            traci.vehicle.setVehicleClass(vid, "emergency")
            traci.vehicle.setColor(vid, (255, 0, 0))
            traci.vehicle.setShapeClass(vid, "delivery")
            traci.vehicle.setSpeedFactor(vid, 1.5)
            
        except Exception as e:
            # If it fails, just ignore it and keep training
            # print(f"Spawn Error: {e}") 
            pass

def get_state():
    """Reads sensors from Hardcoded Lanes"""
    queues = []
    ambs = []
    for direction in ["N", "E", "S", "W"]:
        q = 0
        a = 0
        for lane in LANES[direction]:
            try:
                q += traci.lane.getLastStepHaltingNumber(lane)
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    try:
                        if "amb" in v or traci.vehicle.getVehicleClass(v) == "emergency": a = 1
                    except: pass
            except: pass
        queues.append(min(q / 20.0, 1.0)) 
        ambs.append(a)
    return np.array(queues + ambs)

# --- TRAINING LOOP ---
print("🚀 Starting RAKSHAK 4-WAY Training (Crash-Proof)...")

for e in range(EPISODES):
    try:
        try: traci.close()
        except: pass
        time.sleep(1)
        traci.start(["sumo", "-c", "sumo_intersection/cfg/data2_simulation.sumocfg", "--no-warnings"])
        
        state = get_state()
        total_reward = 0
        step = 0
        
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            traci.simulationStep()
            inject_ambulance(step) 
            
            action = agent.act(state)
            
            target_phase = PHASE_MAP[action]
            current_phase = traci.trafficlight.getPhase("J")
            
            if current_phase != target_phase:
                traci.trafficlight.setPhase("J", target_phase)

            next_state = get_state()
            
            # --- TEACHER LOGIC ---
            # 0=N, 1=E, 2=S, 3=W
            amb_locs = [next_state[4], next_state[5], next_state[6], next_state[7]]
            queues = [next_state[0], next_state[1], next_state[2], next_state[3]]
            
            correct_action = -1
            
            # Priority 1: Ambulance
            if any(amb_locs):
                for i in range(4):
                    if amb_locs[i] > 0:
                        correct_action = i
                        break
            
            # Priority 2: Longest Queue
            else:
                correct_action = np.argmax(queues)
                if queues[correct_action] < 0.1: 
                    correct_action = action # Anti-flicker

            if action == correct_action: reward = 100
            else: reward = -100

            agent.remember(state, action, reward, next_state, False)
            state = next_state
            total_reward += reward
            step += 1
            
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
        
        traci.close()
        agent.update_target_model()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        print(f"Episode {e+1}/{EPISODES} | Score: {total_reward} | Epsilon: {agent.epsilon:.2f}")
        agent.save("rakshak_final_brain_v2.pth")

    except Exception as e_msg:
        print(f"⚠️ Retry: {e_msg}")
        try: traci.close()
        except: pass

print("✅ TRAINING COMPLETE.")