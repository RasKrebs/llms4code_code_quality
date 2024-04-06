from utils.linter import LingtingReport
from utils.radon_generator import RadonAnalyzer
from utils.results import QualityReport
from utils.bandit_eval import Bandit
from utils.memory_profiler_generator import MemoryProfilerScriptGenerator
import glob
import os
import sys
import yaml

# List of folders to be excluded from analysis
folders_not_to_visit = ["archived", "utils", "app_domain_template", "data"]

# List of algorithms
algorithms = [x for x in glob.glob("*") if x not in folders_not_to_visit and os.path.isdir(x)]

# Load yaml with execute statement
with open(".evaluate_config.yml", 'r') as stream:
    config = yaml.safe_load(stream)


class MuteOutput:
    def __init__(self, mute=True):
        self.mute = mute
        self._stdout = sys.stdout

    def __enter__(self):
        if self.mute:
            sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        
load_file = lambda file: open(file, "r").read()

resource_monitor_script = load_file('utils/resource_monitor.py')

data_line = "# --- DATA HERE ---"
main_line = "# --- MAIN CODE ---"
execute_line = "# --- EXECUTE HERE ---"

# Loop through each algorithm
for algorithm in algorithms:
    if algorithm not in ["page_rank"]: #["pca", 'huffman', "monte_carlo_simulation"]:
        continue
    print(f"Analyzing: {algorithm} ({algorithms.index(algorithm)+1}/{len(algorithms)})...\n")
    
    # Extracting resource related scritps
    data_script = load_file(f'{algorithm}/data_loader.txt')
    execute_func = load_file(f'{algorithm}/execute_func.txt')
    execute_statement = load_file(f'{algorithm}/execute_statement.txt')
    
    # Initializing QualityReport() Instance
    report = QualityReport()
    
    # Initializing Bandit() Instance
    print("Running bandit on all files...")
    bandit = Bandit(algorithm)
    
    # Filtering the list of llms to only include valid folders
    algorithm_path = glob.glob(f"{algorithm}/*")
    algorithm_folder = [os.path.isdir(folder) for folder in algorithm_path]
    filtered_list = [value + "/" for value, condition in zip(algorithm_path, algorithm_folder) if condition and value.split('/')[-1] not in folders_not_to_visit]

    # Looping through each LLM
    for llm in filtered_list:
        model = llm.split("/")[-2]
        print(f"Analyzing: {model} ({filtered_list.index(llm)+1}/{len(filtered_list)})...\n")
        
        # Initializing LingtingReport() Instance
        print("Running linter... (1/3)")
        #linter = LingtingReport(llm)
        
        # Initializing RadonAnalyzer() Instance
        print("Running radon... (2/3)")
        #radon = RadonAnalyzer(llm)
        
        # Adding Radon Results to QualityReport
        #report([radon.df, linter.df])
        
        # Generate memory profiler results
        print("Generating resource monitoring scripts... (3/3)")
        
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
    
    report(bandit.df)
    
    report.save_results(sheet_name=algorithm)
        
        
