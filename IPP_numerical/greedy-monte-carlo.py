import numpy as np
import random
from itertools import combinations

# Generate a random dataset with m rows and n columns
def generate_data(m, n, p = 0.5):
    # return np.random.randint(2, size=(m, n))
    a = np.zeros((m,n))
    b = np.ones((m, n))
    return np.where(np.random.rand(m, n) < p, b, a)

# Greedy algorithm to select k columns
def greedy_select(data, k):
    m, n = data.shape
    selected_columns = []
    min_ri = m  # Initialize to maximum possible value
    
    for i in range(k):
        best_column = -1  # Initialize to an invalid column index
        for col in range(n):
            if col in selected_columns:
                continue  # Skip already selected columns
            
            # Consider selecting this column
            temp_columns = selected_columns + [col]
            temp_data = data[:, temp_columns]
            
            # Calculate the minimum ri for this selection
            _, counts = np.unique(temp_data, axis=0, return_counts=True)
            temp_min_ri = np.max(counts)
            
            # Update best_column if this column is better
            if temp_min_ri < min_ri:
                min_ri = temp_min_ri
                best_column = col
                
        # Update selected_columns with the best column found in this iteration
        if best_column != -1:
            selected_columns.append(best_column)
    
    return selected_columns, min_ri

# Function to compute optimal k columns
def optimal_select(data, k):
    m, n = data.shape
    min_ri_optimal = m
    optimal_columns = []
    
    # Iterate through all possible combinations of k columns
    for cols in combinations(range(n), k):
        selected_data = data[:, cols]
        
        # Calculate the minimum ri for this selection
        _, counts = np.unique(selected_data, axis=0, return_counts=True)
        temp_min_ri = np.max(counts)
        
        # Update optimal_columns and min_ri_optimal if this selection is better
        if temp_min_ri < min_ri_optimal:
            min_ri_optimal = temp_min_ri
            optimal_columns = list(cols)
    
    return optimal_columns, min_ri_optimal

# Modified Monte Carlo simulation
def monte_carlo_simulation(trials, p, m, n, k):
    min_ri_list_greedy = []
    min_ri_list_optimal = []
    
    for _ in range(trials):
        data = generate_data(p, m, n)
        # print(data)
        
        # Compute greedy value
        col, min_ri_greedy = greedy_select(data, k)
        min_ri_list_greedy.append(min_ri_greedy)
        # print(col, min_ri_greedy)
        
        # Compute optimal value
        col, min_ri_optimal = optimal_select(data, k)
        min_ri_list_optimal.append(min_ri_optimal)
        # print(col, min_ri_optimal)
        
    average_min_ri_greedy = np.mean(min_ri_list_greedy)
    average_min_ri_optimal = np.mean(min_ri_list_optimal)
    
    return average_min_ri_greedy, average_min_ri_optimal

if __name__ == "__main__":
    trials = 100  # Reduced the number of trials due to the computational cost of the optimal method
    p = 0.2 # Probability of one in the table 
    m = 200  # Number of rows
    n = 12  # Number of columns
    k = 4  # Number of columns to select
    
    average_min_ri_greedy, average_min_ri_optimal = monte_carlo_simulation(trials, p, m, n, k)
    print(f"After {trials} trials, the average minimum r_i achieved by the greedy algorithm is {average_min_ri_greedy}")
    print(f"After {trials} trials, the average minimum r_i achieved by the optimal algorithm is {average_min_ri_optimal}")

