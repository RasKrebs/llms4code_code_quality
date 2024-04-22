from utils.linter import LingtingReport
from utils.radon_generator import RadonAnalyzer
from utils.results import QualityReport
from utils.bandit_eval import Bandit
from utils.memory_profiler_generator import MemoryProfilerScriptGenerator
import glob
import pandas as pd
import os
import sys
import numpy as np
import subprocess


# List of folders to be excluded from analysis
folders_not_to_visit = ["archived", "utils", "app_domain_template", "data"]

# List of algorithms
algorithms = [x for x in glob.glob("*") if x not in folders_not_to_visit and os.path.isdir(x)]

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

        # Adding Radon Results to QualityReport
        report([radon.df, linter.df])

        # Generate memory profiler results
        print("Performing Resource Monitoring... (3/3)")

        # Loop through each script in the resource_monitor folder
        for script in os.listdir(os.path.join(llm, 'resource_monitor')):
            # Skip files that are not python files
            if not script.endswith(".py"):
                continue

            print(f"Executing {script}...")

            # Skip potential files ending with .py
            cpu_usage_list = []
            memory_usage_list = []

            # Loop over the resource monitor 5 times to get the maximum values
            for _ in range(5):
                # Get the path of the script
                script_path = os.path.join(llm, 'resource_monitor', script)

                # Modified line to capture output and errors
                result = subprocess.run(['python3', script_path], check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Decode and process the output
                output = result.stdout.decode('utf-8')
                output_list = output.split('\n')
                output_list = [float(x) for x in output_list if x != '']

                # Append the results to the output list
                cpu_usage_list.append(output_list[0])
                memory_usage_list.append(output_list[1])

            max_cpu = max(cpu_usage_list)
            max_memory = max(memory_usage_list)

            # Append the results to the output list

            data_frame_results = [[y, 'psutil', llm.split('/')[-2], x, script.split('_')[0]] for x, y in zip([max_cpu, max_memory], ['cpu','memory_usage'])]

            resource_frame = pd.DataFrame(data_frame_results, columns=['metric', 'framework', 'model','value','prompt'])

            # Add the results to the report
            report(resource_frame)

    report(bandit.df)

    report.save_results(sheet_name=algorithm)
