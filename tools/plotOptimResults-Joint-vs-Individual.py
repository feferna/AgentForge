import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data_from_db(db_path, algorithm_name, limit_trials=None):
    conn = sqlite3.connect(db_path)
    
    # For all algorithms, load trial values from "trial_values" table
    query = "SELECT trial_id AS solution_id, value AS fitness FROM trial_values"
    df = pd.read_sql_query(query, conn)
    df['Trial number'] = df['solution_id']  # Assuming trial starts at 1

    conn.close()
    
    if limit_trials is not None:
        df = df[df['Trial number'] <= limit_trials]  # Limit number of trials to the specified value
    
    df['algorithm'] = algorithm_name
    return df

# Load data from BO-NoNAS to get the maximum number of trials for other algorithms
df_bo_nonas = load_data_from_db('BO-NoNAS.db', 'Optimization without NAS', limit_trials=400)

# Determine the maximum number of trials from BO-NoNAS database
max_trials = df_bo_nonas['Trial number'].max()

# Load data from BO-Joint.db and limit the number of trials to max_trials from BO-NoNAS
df_bo = load_data_from_db('BO-Joint.db', 'Joint Optimization', limit_trials=max_trials)

# Load data from BO-OnlyNAS.db and limit the number of trials to max_trials from BO-NoNAS
df_only_nas = load_data_from_db('BO-OnlyNAS.db', 'Only NAS Optimization', limit_trials=max_trials)

# Combine the data into a single DataFrame
df = pd.concat([df_bo_nonas, df_bo, df_only_nas])

# Compute cumulative max fitness
df.sort_values('Trial number', inplace=True)
df['Max Fitness'] = df.groupby('algorithm')['fitness'].cummax()

# Create new column to identify the trials in which a new maximum is found
df['new_max_found'] = df.groupby('algorithm')['Max Fitness'].diff() != 0

# Plot the data
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))  # Adjusted figure size for a single column

# Extract unique algorithm names
algorithms = df["algorithm"].unique()

for algorithm in algorithms:
    print(algorithm)
    # Plot maximum fitness line per algorithm
    subset_df = df[df["algorithm"] == algorithm]
    plt.plot(subset_df['Trial number'], subset_df['Max Fitness'], label=algorithm)

    # Add markers at the positions where a further maximum is found
    max_positions = subset_df[subset_df['new_max_found']]
    plt.scatter(max_positions['Trial number'], max_positions['Max Fitness'])

    # Highlight the final maximum Mean Reward
    final_max = subset_df['Max Fitness'].max()
    final_trial = subset_df['Trial number'].max()
    
    if algorithm == 'Joint Optimization':
        plt.text(final_trial - 10, final_max - 25, f'{final_max:.2f}', 
                 color='black', fontsize=14, ha='center')
    else:
        plt.text(final_trial - 10, final_max + 10, f'{final_max:.2f}', 
                 color='black', fontsize=14, ha='center')

plt.xticks(fontsize=14)  # Increase font size for x-axis values
plt.yticks(fontsize=14)  # Increase font size for y-axis values

# Increased font sizes for labels and legend
plt.xlabel('Trial Number', fontsize=16)  # Increased font size for x-axis label
plt.ylabel('Mean Reward', fontsize=16)  # Increased font size for y-axis label

plt.legend(fontsize=14)  # Increased font size for legend
plt.grid(True)
plt.tight_layout()  # Adjusts the plot to fit within the figure area

# Save the figure as a PDF and PNG
plt.savefig('bo_comparison_plot.pdf', format='pdf')
plt.savefig('bo_comparison_plot.png', format='png')

# Show the plot
plt.show()

# Plot with y-axis between 100 and 200
plt.figure(figsize=(6, 4))  # Adjusted figure size for a single column

for algorithm in algorithms:
    # Plot maximum fitness line per algorithm
    subset_df = df[df["algorithm"] == algorithm]
    plt.plot(subset_df['Trial number'], subset_df['Max Fitness'], label=algorithm)

    # Add markers at the positions where a further maximum is found
    max_positions = subset_df[subset_df['new_max_found']]
    plt.scatter(max_positions['Trial number'], max_positions['Max Fitness'])