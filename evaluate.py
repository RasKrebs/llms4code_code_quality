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
        
        
# Loop through each algorithm
for algorithm in algorithms:
    print(f"Analyzing: {algorithm} ({algorithms.index(algorithm)+1}/{len(algorithms)})...\n")
    
    # Get the execute statement
    execute = config[algorithm]
    
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
        linter = LingtingReport(llm)
        
        # Initializing RadonAnalyzer() Instance
        print("Running radon... (2/3)")
        radon = RadonAnalyzer(llm)
        
        # Generate memory profiler results
        print("Generating memory profiler scripts... (3/3)")
        profiler = MemoryProfilerScriptGenerator(llm,
                                                 execute_statement=execute)
        
        
        # Adding Radon Results to QualityReport
        report([radon.df, linter.df])
    
    report(bandit.df)
    
    report.save_results(sheet_name=algorithm)
    
    