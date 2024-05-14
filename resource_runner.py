import subprocess
import os
import glob

folders_not_to_visit = ["archived", "utils", "app_domain_template", "data"]
algorithms = [x for x in glob.glob("*") if x not in folders_not_to_visit and os.path.isdir(x)]
load_file = lambda file: open(file, "r").read()
resource_monitor_script = load_file('utils/resource_monitor.py')
data_line = "# --- DATA HERE ---"
main_line = "# --- MAIN CODE ---"
execute_line = "# --- EXECUTE HERE ---"

# Loop through each algorithm
for algorithm in algorithms:
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


######## SCRIPT TO AUTOMATICALLY GENERATE RESOURCE MONITOR SCRIPTS
        # Currently commented out to not overwrite the existing resource monitor scripts
        # os.makedirs(os.path.join(llm, 'resource_monitor'), exist_ok=True)
        #
        # # Loop through each file in the LLM
        # for file in os.listdir(llm):
        #     # Skip files that are not python files
        #     if file.endswith(".py"):
        #         # Extract the main code
        #         main_code = load_file(f'{llm}/{file}')
        #
        #         pre_data = resource_monitor_script.split(data_line)[0] + '\n'
        #         post_data = resource_monitor_script.split(data_line)[1].split(main_line)[0] + '\n'
        #         post_execute = resource_monitor_script.split(execute_line)[1] + '\n'
        #         post_execute = post_execute.replace('    output = execute(x)', execute_statement)
        #         code = pre_data + data_script + post_data + main_code + execute_func + post_execute
        #
        #         # Write the resource monitor script
        #         with open(os.path.join(llm, 'resource_monitor', file.replace('.py', '') + '_resource_version.py'), "w") as f:
        #             f.write(code)
        #     else:
        #         continue

