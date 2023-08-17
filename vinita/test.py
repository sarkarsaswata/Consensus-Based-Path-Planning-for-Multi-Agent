# Time comparision between task allocation with collision avoidance and without collision avoidance 100 simulation with 100 iterations



import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import LineString, Point
from Environment import Spown, animate, update, reached, clear
import time
import json
plt.ion()


def generate_agents_and_tasks(n, max_coordinate=100):
    """
    Generate n random agents and n random tasks with random coordinates between 0 and max_coordinate.

    Args:
        n (int): The number of agents and tasks to generate.
        max_coordinate (int): The maximum value of the x and y coordinates. Default is 100.

    Returns:
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.
    """
    # # np.random.seed(90)
    # agents = np.random.rand(n, 2) ma * max_coordinate
    # tasks = np.random.rand(n, 2) *x_coordinate
    # [-2.0, 2.5], [-1.5, 5.5]
    agents_x = np.random.uniform(-2.0, 2.5, n)
    agents_y = np.random.uniform(-1.5, 5.5, n)
    agents = np.column_stack((agents_x, agents_y))

    tasks_x = np.random.uniform(-2.0, 2.5, n)
    tasks_y = np.random.uniform(-1.5, 5.5, n)
    tasks = np.column_stack((tasks_x, tasks_y))
    return agents, tasks


def generate_cost_matrix(n, agents, tasks):
    """
    Generate a cost matrix for assigning agents to tasks based on the inverse Euclidean distance.

    Args:
        n (int): The number of agents and tasks.
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.

    Returns:
        C (np.ndarray): An array of shape (n, n) representing the cost of assigning agent i to task j.
    """
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Compute the inverse Euclidean distance as the cost between the agent and the task
            C[i][j] = 1.0 / np.linalg.norm(agents[i] - tasks[j])
   
    return C


def compute_total_cost(i, j, agents, tasks):
    """Compute the total cost of the assignment."""
    cost = np.linalg.norm(agents[i] - tasks[j])
    return cost


def visualize_assignments(agents, tasks, assignment):
    """
    Visualize the assignment of agents to tasks.

    Args:
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.
        assignment (np.ndarray): An array of shape (n,) representing the assignment of agents to tasks.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(agents)):
        ax.scatter(agents[i][0], agents[i][1], c='b')  # Plot the agent
        ax.text(agents[i][0], agents[i][1], 'A{}'.format(i))  # Label the agent

    for i in range(len(tasks)):
        ax.scatter(tasks[i][0], tasks[i][1], c='r')  # Plot the task
        ax.text(tasks[i][0], tasks[i][1], 'T{}'.format(i))  # Label the task

    for i in range(len(assignment)):
        task = assignment[i]
        if task >= 0:
            # Plot the line between the agent and the task
            ax.plot([agents[i][0], tasks[task][0]], [agents[i][1], tasks[task][1]], c='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def task_assignment(agents, tasks, n, C):
    """
    Compute the task assignment.

    Args:
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.
        n (int): The number of agents and tasks.
        C (np.ndarray): An array of shape (n, n) representing the cost of assigning agent i to task j.

    Returns:
        X (np.ndarray): An array of shape (n, n) representing the task assignment matrix.
    """
    X = np.zeros((len(agents), len(tasks)))
    Y = np.zeros((len(agents), len(tasks)))
    H = np.zeros((len(agents), len(tasks)))
    G = np.ones((len(agents), len(agents)))

    for agent in range(n):
        # Auction Algorithm
        if np.sum(X[agent]) < 1:  # if agent has not been assigned a task
            for task in range(n):
                if np.sum(X[:, task]) > 0:
                    # skip tasks that have already been assigned to another agent
                    continue
                H[agent][task] = 1 if C[agent][task] > Y[agent][task] else 0
            if np.sum(H[agent]) != 0:  # if agent has not been assigned a task
                dot = H[agent] * C[agent]  # dot product of the agent's row in H and the agent's column in C
                argmax_j = np.argmax(dot)  # find the task with the highest value for the agent
                X[agent][argmax_j] = 1  # assign the task to the agent
                Y[agent][argmax_j] = C[agent][argmax_j]  # update the agent's task value in Y

        # Consensus Algorithm
            for task in range(n):  # for each task
                dot = G[agent] * Y[:, task]  # dot product of the agent's row in G and the task's column in Y
                Y[agent][task] = np.max(dot)  # update the agent's task value in Y

            dot = G[agent] * Y[:, argmax_j]  # dot product of the agent's row in G and the task's column in Y
            argmax_z = np.argmax(dot)  # find the agent with the highest value for the task

            if argmax_z != agent:  # if the agent with the highest value for the task is not the current agent
                X[agent][argmax_j] = 0  # remove the task from the current agent

    return X




def resolve_collisions(X, agents, tasks, threshold_radius):
    """
    Resolve collisions between agents by reassigning tasks.

    Args:
        X (np.ndarray): An array of shape (n, n) representing the task assignment matrix.
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.

    Returns:
        X (np.ndarray): An array of shape (n, n) representing the updated task assignment matrix.
    """
    no_of_collision = 0
    
    X_temp = np.zeros_like(X)
    X_copy = X.copy() 
    obstacle1 = []
    obstacle2 = []
    # print(np.any(np.not_equal(X_temp, X_copy)))
    while np.any(np.not_equal(X_temp, X_copy)):
        X_copy = X.copy() 
        for i in range(n):
            for j in range(n):
                if X[i][j] == 1:
                    agent_loc = Point(agents[i])
                    task_loc = Point(tasks[j])
                    line = LineString([agent_loc, task_loc])

                    for i2 in range(i+1, n):
                        for j2 in range(n):
                            if X[i2][j2] == 1:
                                agent_loc2 = Point(agents[i2])
                                task_loc2 = Point(tasks[j2])
                                line2 = LineString([agent_loc2, task_loc2])

                                if line.intersects(line2):
                                    inter = line.intersection(line2)
                                    inter_point = [inter.x, inter.y]
                                    dist1 = agent_loc.distance(inter)
                                    dist2 = agent_loc2.distance(inter)

                                    if abs(dist1 - dist2) < threshold_radius:
                                        cost_old = compute_total_cost(i, j, agents, tasks) + compute_total_cost(i2, j2, agents, tasks)
                                        cost_new = compute_total_cost(i, j2, agents, tasks) + compute_total_cost(i2, j, agents, tasks)
                                        
                                        if ((cost_old - cost_new)/cost_old)*100 >= 5: 
                                            # print("Switching the task between agents")
                                            obstacle1.append([i, i2,inter_point])
                                            no_of_collision += 1
                                            X[i][j2] = 1
                                            X[i2][j] = 1
                                            X[i][j] = 0
                                            X[i2][j2] = 0

                                            break

                                        else:
                                            # agent_task_id.append([i, j])
                                            # agent_task_id.append([i2, j2])
                                            obstacle1.append([i, i2,inter_point])
                                            obstacle2.append([i, i2,inter_point])
                                            # print(" NOT Switching the task between agents")
        X_temp = X.copy()                            
    # print("swapping",no_of_collision)
    # print(" APF ",len(obstacle))
    swapp = no_of_collision
    collision = (swapp + len(obstacle2)) #total avoidance
    print("*************",collision)                             
    return X, obstacle1,obstacle2 , swapp, collision


def reset():
    clear()
    # print("inside reset")
    n, threshold_radius, agents, tasks, X, X_up, obs1, obs2, agent_task, swapp, agent1_task1, total_col = calculation()
    spown = Spown(agents, tasks, agent_task, obs, threshold_radius)
    return spown
        
    
n = 50
threshold_radius = 0.5

def calculation():
    
    agents, tasks = generate_agents_and_tasks(n)
    C = generate_cost_matrix(n, agents, tasks)
    X = task_assignment(agents, tasks, n, C)
    X_c = X.copy()
    # assignment = np.argmax(X, axis=1)
    # print(type(X), X.shape)
    # visualize_assignments(agents, tasks, assignment)
    X_up, obs1, obs2, swapp, total_col = resolve_collisions(X, agents, tasks, threshold_radius)
    # assignment_up = np.argmax(X_up, axis=1)
    # visualize_assignments(agents, tasks, assignment_up)
    
    agent1_task1 = []
    for id, i in enumerate(X_c):
        count = -1
        for j in i:
            count += 1
            if j == 1:
                agent1_task1.append([id, count])
    
    agent_task = []
    for id, i in enumerate(X_up):
        count = -1
        for j in i:
            count += 1
            if j == 1:
                agent_task.append([id, count])

    return n, threshold_radius, agents, tasks, X_c, X_up, obs1, obs2, agent_task, swapp, agent1_task1, total_col


def simulation(spown, n):
        count = 0
        time = []
        for i in np.arange(0, 50, 0.1):
            
            spown, time = reached(spown, time)
           
            if len(spown) >= 1:
                for one in spown:
                    update(one)
                    
            else:
                clear()
                break
            animate(spown)
            
        # print(time)
        return time
        



data_collection = []

for i in range(100):  #no of simulation
    # n += 2
    # for j in range(100):
        n, threshold_radius, agents, tasks, X, X_up, obs1, obs2, agent_task, swapp, agent1_task1, total_col = calculation()
    
        print("n",n)
        agent_pos = []
        task_pos = []
        print("swapping", swapp, i)
        for p in agents:
            agent_pos.append(p.tolist())
        # print(agent_pos, type(agent_pos))
        for q in agents:
            task_pos.append(q.tolist())
        time_w = []
        time_wo = []
        # wo collision avoidance
        for j in range(1):   #no of iteration
            spown = Spown(agents, tasks, agent1_task1, obs = [], threshold_radius = 5) 
            print("Without CA")
            time_wo = simulation(spown, n)
            # time_wo.append(time1)
            # with collision avoidance
            spown = Spown(agents, tasks, agent_task, obs2, threshold_radius) 
            print("with CA")
            time_w = simulation(spown, n)
            # time_w.append(time2)
        
   
        data_g = {'n' : n, 'agent_pos' : agent_pos, 'task_pos' : task_pos, 'at_pair' : agent_task,'t1' : time_wo, 't2' : time_w, 'col' : total_col, 'swappping' : swapp}
        # data_g = {'n' : n,'t1' : time_wo, 't2' : time_w}
        data_collection.append(data_g)


with open('data9.json', 'w') as file:
    json.dump(data_collection, file, indent=4)  # Indent for readability
