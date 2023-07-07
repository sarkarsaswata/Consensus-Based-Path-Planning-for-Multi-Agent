import math

def calculate_direction():
    pass


def repulsion(x, y, g_x, g_y, obstacle_pose, r):
    direction = math.atan2(g_y - y, g_x - x)
    if obstacle_pose != None:
        if math.dist((x, y), (obstacle_pose[0], obstacle_pose[1])) <= r:
            direction += math.pi / 2

    return direction


# def adjust_agent_direction(agent_direction, obstacle_center, obstacle_radius):
#     distance = math.sqrt((agent_direction[0] - obstacle_center[0])**2 + (agent_direction[1] - obstacle_center[1])**2)
#     comp = agent_direction
#     if distance <= obstacle_radius:
#         # Conflict with the obstacle, adjust agent direction
#         agent_direction += math.pi/2
#     if comp != agent_direction:
#         print("repulsion")
#     else:
#         print("not happening")
#     return agent_direction


# # current_x, current_y = 66, 56
# # goal_x, goal_y = 
# # obstacle_center = [18, 26]
# # obstacle_radius = 10


# def calculate_direction_vector(current_x, current_y, goal_x, goal_y):
#     delta_x = goal_x - current_x
#     delta_y = goal_y - current_y

#     magnitude = math.sqrt(delta_x**2 + delta_y**2)
    
#     if magnitude == 0:
#         return 0, 0

#     direction_x = delta_x / magnitude
#     direction_y = delta_y / magnitude

    # return [direction_x, direction_y]


# agent_direction = calculate_direction_vector(current_x, current_y, goal_x, goal_y)
# print(agent_direction)
# modified_direction = adjust_agent_direction(agent_direction, obstacle_center, obstacle_radius)
# print(modified_direction)
