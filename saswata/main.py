import numpy as np

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

def initialize_task_matrix(n):
    # Initialize an n x n task allocation matrix X with zeros.
    return np.zeros((n, n), dtype=int)

def allocate_tasks(X, agents, C):
    n = X.shape[0]
    task_list = list(range(n))  # Initialize a list of task IDs.
    task_allocation = {}  # Dictionary to store agent-task assignments.

    while task_list and agents:
        for agent in agents:
            # Calculate the sum of task values for each column.
            column_sums = X.sum(axis=0)

            # Filter out tasks that have already been allocated.
            available_tasks = [task for task in task_list if column_sums[task] == 0]

            if not available_tasks:
                break  # No available tasks for this agent.

            # Choose the task with the highest value for this agent.
            selected_task = max(available_tasks, key=lambda task: C[agent][task])

            # Update task allocation dictionary.
            task_allocation[agent] = selected_task

            # Remove the selected task from the task list.
            task_list.remove(selected_task)

            # Update the task matrix X.
            X[agent] = 0
            X[agent][selected_task] = 1

            # Remove this agent from the list of agents.
            agents.remove(agent)

    return task_allocation

def consensus_phase(X, task_allocation):
    for agent, task in task_allocation.items():
        # Check if any other agent has chosen the same task.
        conflicting_agents = [a for a, t in task_allocation.items() if t == task]

        if len(conflicting_agents) > 1:
            # Find the agent with the lowest value for the task.
            lowest_value_agent = min(conflicting_agents, key=lambda a: X[a][task])

            for agent in conflicting_agents:
                if agent != lowest_value_agent:
                    # Assign the task to the agent with the lowest value.
                    task_allocation[agent] = -1  # Mark agent as unallocated (-1).

    return task_allocation

def main():
    n = 5  # Number of tasks and agents (adjust as needed).
    agents, tasks = generate_agents_and_tasks(n)
    C = generate_cost_matrix(n, agents, tasks)
    X = initialize_task_matrix(n)

    task_allocation = allocate_tasks(X, list(range(n)), C)
    print("Initial Task Allocation:")
    print(task_allocation)

    task_allocation = consensus_phase(X, task_allocation)
    print("\nTask Allocation After Consensus Phase:")
    print(task_allocation)

if __name__ == "__main__":
    main()
