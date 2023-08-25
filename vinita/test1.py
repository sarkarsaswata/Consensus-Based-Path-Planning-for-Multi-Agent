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


def auction_algorithm(agents, tasks, n, C):
    """
    Perform the Auction Algorithm for task assignment.

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

    for agent in range(n):
        if np.sum(X[agent]) < 1:  # if agent has not been assigned a task
            for task in range(n):
                if np.sum(X[:, task]) > 0:
                    # skip tasks that have already been assigned to another agent
                    continue
                Y[agent][task] = C[agent][task]

            argmax_j = np.argmax(Y[agent])  # find the task with the highest value for the agent
            X[agent][argmax_j] = 1  # assign the task to the agent

    return X


def consensus_algorithm(X, agents, tasks, n, C):
    """
    Perform the Consensus Algorithm for task assignment.

    Args:
        X (np.ndarray): The output of the Auction Algorithm, representing the initial task assignments.
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.
        n (int): The number of agents and tasks.
        C (np.ndarray): An array of shape (n, n) representing the cost of assigning agent i to task j.

    Returns:
        X (np.ndarray): An array of shape (n, n) representing the task assignment matrix.
    """
    Y = np.zeros((len(agents), len(tasks)))
    G = np.ones((len(agents), len(agents)))

    for agent in range(n):
        for task in range(n):  # for each task
            dot = G[agent] * Y[:, task]  # dot product of the agent's row in G and the task's column in Y
            Y[agent][task] = np.max(dot)  # update the agent's task value in Y

    return X  # Return the initial assignment matrix (Auction result)


def task_assignment(agents, tasks, n, C):
    """
    Compute the task assignment using the Auction Algorithm and the Consensus Algorithm.

    Args:
        agents (np.ndarray): An array of shape (n, 2) representing n agents, with each agent represented by a 2D coordinate.
        tasks (np.ndarray): An array of shape (n, 2) representing n tasks, with each task represented by a 2D coordinate.
        n (int): The number of agents and tasks.
        C (np.ndarray): An array of shape (n, n) representing the cost of assigning agent i to task j.

    Returns:
        X (np.ndarray): An array of shape (n, n) representing the task assignment matrix.
    """
    X = auction_algorithm(agents, tasks, n, C)
    X = consensus_algorithm(X, agents, tasks, n, C)
    return X
