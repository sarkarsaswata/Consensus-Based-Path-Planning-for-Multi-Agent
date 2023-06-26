import math

def adjust_agent_direction(agent_direction, obstacle_center, obstacle_radius):
    distance = math.sqrt((agent_direction[0] - obstacle_center[0])**2 + (agent_direction[1] - obstacle_center[1])**2)
    
    if distance <= obstacle_radius:
        # Conflict with the obstacle, adjust agent direction
        agent_direction += math.pi/2
    
    return agent_direction
