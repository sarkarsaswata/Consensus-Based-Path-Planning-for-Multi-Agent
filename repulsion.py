import math

def repulsion(x, y, g_x, g_y, obstacle_pose, r):
    direction = math.atan2(g_y - y, g_x - x)
    e = 0.1
    if obstacle_pose != None:
        if math.dist((x, y), (obstacle_pose[0], obstacle_pose[1])) <= r:
            angle = math.atan2(g_y - y, g_x - x)
            if 0 <= angle <= (math.pi - e):
                direction += math.pi / 2

            print("angle", angle)
    return direction



# def repulsion(x, y, g_x, g_y, obstacle_pose, r):
#     direction = math.atan2(g_y - y, g_x - x)
#     if obstacle_pose != None:
#         if math.dist((x, y), (obstacle_pose[0], obstacle_pose[1])) <= r:
#             direction += math.pi / 2

#     return direction


 