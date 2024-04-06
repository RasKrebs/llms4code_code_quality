import subprocess
import os
import glob

folders_not_to_visit = ["archived", "utils", "app_domain_template", "data"]
algorithms = [x for x in glob.glob("*") if x not in folders_not_to_visit and os.path.isdir(x)]


# Loop through each algorithm
for algorithm in algorithms:
    if algorithm not in ['convolution']: #,'monte_carlo_simulation','huffman']:
        continue
    # Filtering the list of llms to only include valid folders
    algorithm_path = glob.glob(f"{algorithm}/*")
    algorithm_folder = [os.path.isdir(folder) for folder in algorithm_path]
    filtered_list = [value + "/" for value, condition in zip(algorithm_path, algorithm_folder) if condition and value.split('/')[-1] not in folders_not_to_visit]
    
    # Looping through each LLM
    for llm in filtered_list:
        for script in os.listdir(os.path.join(llm, 'resource_monitor')):
            if not script.endswith(".py"):
                continue
            script_path = os.path.join(llm, 'resource_monitor', script)
            print(f"Executing {script_path}...")
            try:
                subprocess.run(['python3', script_path], check=True, timeout=20)

            except subprocess.CalledProcessError:
                print(f"Error in {script_path}")
