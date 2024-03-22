import subprocess
import pandas as pd
import json
import os

class Bandit:
    
    def __init__(self, algorithm_path:str ):
        """Initializes the Bandit class, which is used to run bandit on the algorithms. Methods generate a temp json file while executing, which is deleted after the execution is complete.

        Args:
            algorithm_path (str): Takes the path of the algorithm to be analyzed.
        """
        # Variables
        self.path:str = algorithm_path
        self.file_name:str = "_tmp_bandit_results.json"
        self.tmp = os.path.join(self.path, 'tmp')
        self.output_json = os.path.join(self.tmp, self.file_name)
        self.framework = "bandit"
        self.df = pd.DataFrame(columns=['metric', 'framework', 'model','value','prompt'])
        
        # File names and substrings to skip while analyzing
        self.skip = ['archived', 'flake8_summarizer', 'memory_usage', 'utils', 'write_to_file', '_total']
        
        # Run the report
        self._run()
                
    def _run(self):
        """Runs bandit on the algorithm.
        """
        # Bandit command
        command = f"bandit -f json -o {self.output_json} -ll -r {self.path} --exclude archived --exclude memory_usage"
        
        # Create temporary folder
        self._create_tmp()
        
        # Runs bandit
        _ = subprocess.run(command, capture_output=True, text=True, shell=True)
        
        # Load the results
        results = self._load_results()
        
        # Wrangle the results
        self._wrangle_results(results)
        
        # Delete temporary folder
        self._del_tmp()
        
    def _create_tmp(self):
        """Creates a temporary folder to store json file which is used for the results of the bandit analysis.
        """
        # Create temporary folder
        if not os.path.exists(self.tmp):
            os.mkdir(self.tmp)
            
    def _load_results(self):
        """Loads the results of the bandit analysis.
        """
        # Load the results
        with open(self.output_json, "r") as file:
            # json file
            results = json.load(file)
        return results['metrics']
    
    
    def _wrangle_results(self, results):
        """Wrangles the results of the bandit analysis to a pandas DataFrame.
        """        
        # Loop through the results
        for result in results.keys():
            if any([x in result for x in self.skip]):
                continue
            
            # Extract the model and prompt
            model = result.split('/')[-2]
            prompt = result.split('/')[-1].replace('.py', '')
            
            # Extract the severity
            try: 
                # Extract the severity
                high_severity = results[result]['SEVERITY.HIGH']
                med_severity = results[result]['SEVERITY.MEDIUM']
                low_severity = results[result]['SEVERITY.LOW']
                
                # Create the data
                data = [[metric, self.framework, model, value, prompt] for metric, value in zip(['security_high', 'security_medium', 'security_low'], [high_severity, med_severity, low_severity])]
                out = pd.DataFrame(data, columns=['metric', 'framework', 'model','value','prompt'])
                
                # Append to the main dataframe
                self.df = pd.concat([self.df, out])
            except: 
                pass
        
    def _del_tmp_json(self):
        """Deletes the json file.
        """
        # Delete temporary folder
        if os.path.exists(self.output_json):
            os.remove(self.output_json)            

    def _del_tmp(self):
        """Deletes the temporary folder and the json file.
        """
        # Delete temporary json file
        self._del_tmp_json()
        
        # Delete temporary folder
        if os.path.exists(self.tmp):
            os.rmdir(self.tmp)
        
        