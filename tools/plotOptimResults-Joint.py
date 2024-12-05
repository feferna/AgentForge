import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data_from_db(db_path, algorithm_name, has_generation):
    conn = sqlite3.connect(db_path)
    
    if has_generation:
        # For PSO: Load from "solution_history" and compute trial number
        query = "SELECT generation, solution_id, fitness FROM solution_history"
        df = pd.read_sql_query(query, conn)
        df['Trial number'] = df['generation'] * solutions_per_generation + df['solution_id']
    else:
        # For BO and Random Search: Load from "trial_values"
        query = "SELECT trial_id AS solution_id, value AS fitness FROM trial_values"
        df = pd.read_sql_query(query, conn)
        df['Trial number'] = df['solution_id'] # Assuming trial starts at 1

    conn.close()
    df['algorithm'] = algorithm_name
    return df

# Define the number of solutions per generation for PSO
solutions_per_generation = 20

# Load data from each database
df_pso = load_data_from_db('PSO-Joint.db', 'PSO', has_generation=True)
df_bo = load_data_from_db('BO-Joint.db', 'Bayesian Optimization', has_generation=False)
df_rs = load_data_from_db('RandomSearch-Joint.db', 'Random Search', has_generation=False)

# Combine all data into a single DataFrame
df = pd.concat([df_pso, df_bo, df_rs])

# Compute cumulative max fitness
df.sort_values('Trial number', inplace=True)
df['Max Fitness'] = df.groupby('algorithm')['fitness'].cummax()

# Create new column to identify the trials in which a new maximum is found
df['new_max_found'] = df.groupby('algorithm')['Max Fitness'].diff() != 0

# Limit Bayesian Optimization and Random Search results to a maximum of 400 trials
df = df[(df['algorithm'] != 'PSO') & (df['Trial number'] <= 400) | (df['algorithm'] == 'PSO')]

# Determine final maximum Mean Reward for each algorithm
final_max = df.groupby('algorithm').agg({
    'Max Fitness': 'max',
    'Trial number': 'last'
}).reset_index()

# Plot the data
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))  # Adjusted figure size for a single column

# Extract unique algorithm names
algorithms = df["algorithm"].unique()

for algorithm in algorithms:
    # Plot maximum fitness line per algorithm
    subset_df = df[df["algorithm"] == algorithm]
    plt.plot(subset_df['Trial number'], subset_df['Max Fitness'], label=algorithm)

    # Add markers at the positions where a further maximum is found
    max_positions = subset_df[subset_df['new_max_found']]
    plt.scatter(max_positions['Trial number'], max_positions['Max Fitness'])

    # Highlight the final maximum Mean Reward
    final_row = final_max[final_max['algorithm'] == algorithm]
    plt.text(final_row['Trial number'].values[0] - 20, final_row['Max Fitness'].values[0] - 20,
             f'{final_row["Max Fitness"].values[0]:.2f}', 
             color='black', fontsize=14, ha='center')

plt.xticks(fontsize=14)  # Increase font size for x-axis values
plt.yticks(fontsize=14)  # Increase font size for y-axis values

# Increased font sizes for labels and legend
plt.xlabel('Trial Number', fontsize=16)  # Increased font size for x-axis label
plt.ylabel('Mean Reward', fontsize=16)  # Increased font size for y-axis label
#plt.title('Comparison of Mean Reward over Trials for Different Algorithms', fontsize=18)  # Increased font size for title

plt.legend(fontsize=14)  # Increased font size for legend
plt.grid(True)
plt.tight_layout()  # Adjusts the plot to fit within the figure area

# Save the figure as a PDF
plt.savefig('comparison_plot.pdf', format='pdf')
plt.savefig('comparison_plot.png', format='png')

# Show the plot
plt.show()