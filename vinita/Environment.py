import numpy as np
import math
import matplotlib.pyplot as plt
from repulsion import repulsion 


fig, ax = plt.subplots()

class object:
    def __init__(self):
            self.r = 0.1
            self.id1, self.id2 = 0, 0 #id1 is agents id, id2 is task id alloted to agent having id1
            # self.agent_pose = agents[i[0]]
            self.x1, self.y1 = 0, 0 #agent's position
            self.x1_temp, self.y1_temp = 0 ,0
            # print(self.x1, self.y1)
            # self.goal_pose = tasks[i[1]] 
            self.x2, self.y2 = 0, 0 #task's position
            self.body1 = None #to store the value for agent
            self.body2 = None #to store the value for task
            self.speed = 0.2
            self.obstacle = None
            self.threshold_radius = 0
            self.trajectory = []
            self.collision = []
            self.time = []

def Spown(agents, tasks, agent_task, obs, threshold_radius):
    spown = []
    for i in agent_task:
        ag = object()
        spown.append(ag)
        ag.id1, ag.id2 = i
        if obs != []:
            for list1 in obs:
                if ag.id1 in (list1[0], list1[1]):
                    ag.obstacle = list1[2]
        # print(ag.obstacle, ag.id1)
        # print("inside Spown")
        # # print(agents[i[0]], tasks[i[1]] )
        ag.x1, ag.y1 = agents[i[0]]
        ag.x2, ag.y2 = tasks[i[1]] 
        # print(ag.x1, ag.y1)
        ag.body1 = plt.Circle((ag.x1, ag.y1), ag.r, facecolor = 'blue')
        ax.add_patch(ag.body1)
        ag.body2 = plt.Circle((ag.x2, ag.y2), ag.r, facecolor = 'red')
        ax.add_patch(ag.body2)
        ag.threshold_radius = threshold_radius
    ax.set_xlim(-2.0, 2.5)
    ax.set_ylim(-1.5, 5.5)
    #[-2.0, 2.5], [-1.5, 5.5]
    ax.set_aspect('equal')
    plt.show()
        
    
    return spown


def reached(spown, time):
    ls_agents = []
    for i in spown:
        if math.dist((i.x1, i.y1), (i.x2, i.y2)) <= 0.1:
            time.append([i.id1, len(i.time)])
            i.time = []
            continue
        else:
            ls_agents.append(i)
        
    return ls_agents, time


def clear():
    ax.clear()
    



def dynamics(one, agent_direction):
    
    pose = np.array([one.x1, one.y1])
    # # print(one.y1)
    new_pose = (pose[0] + (one.speed*math.cos(agent_direction)), pose[1] + (one.speed*math.sin(agent_direction)))
    # print(new_pose[1] )
    return new_pose



def update(one):
    agent_direction = repulsion(one.x1, one.y1, one.x2, one.y2, one.obstacle, one.threshold_radius)
    one.x1_temp, one.y1_temp = dynamics(one, agent_direction)
    one.time.append(1)
    # one.trajectory.append((one.x1_temp, one.y1_temp))
    

def animate(spown):
    com = []
    for t in spown:
        dist = math.dist

    
        t.x1, t.y1 = t.x1_temp, t.y1_temp
        t.body1.set(center=(t.x1,t.y1))
        # trajectory_x, trajectory_y = zip(*t.trajectory)
        # plt.plot(trajectory_x, trajectory_y, color='gray', linewidth=0.5)
        # print("Inside animate")
    plt.pause(0.001)


# def simulation(spown):
#     for i in np.arange(0,100, 0.1):
#         for one in spown:
            
#             update(one)
            
           
#         animate(spown)