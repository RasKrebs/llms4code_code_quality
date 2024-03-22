# Libraries
import glob
import pylint.lint
import pandas as pd


# Class
class LingtingReport:
    def __init__(self, file_path):
        # Path to the Python file
        self.file_path = file_path
        self.lint_config = "utils/.pylintrc"
        
        # Get all files in file_path that end with .py
        # Check if file_path ends with '/'
        if not file_path.endswith('/'):
            file_path += '/'
        self.py_files = glob.glob(file_path + "*.py")

        # Create a dataframe to store the results
        self.df = pd.DataFrame(columns=["metric", "framework", "model", "value"])
        
        # Run Pylint on the files
        self._lint()

    def _generate_frame(self, value, model, prompt, metric):
        # Return the frame
        out = pd.DataFrame({
            "metric": metric,
            "framework": "pylint",
            "model": model,
            "prompt": prompt,
            "value": value}, index=[0])
        
        return out
    
    def _lint(self):
        # Run Pylint on the files
        for file in self.py_files:
            # List the file that is being linted
            print("Linting file: " + '/'.join(file.split("/")[-2:]) + "...")
            model = file.split("/")[-2]
            prompt = file.split("/")[-1].replace(".py", "")
            
            # Run Pylint on the file
            # Create an instance of your custom reporter
            '/utils/.pylintrc'
            results = pylint.lint.Run([(f"--rcfile={self.lint_config}"), file], do_exit=False)
            
            # Extract the score and number of methods
            score = results.linter.stats.__dict__['global_note']
            score = self._generate_frame(value=score,
                                        model=model,
                                        prompt=prompt,
                                        metric="pylint_score")
            
            # Extract the number of methods
            methods = results.linter.stats.__dict__['node_count']['method'] + results.linter.stats.__dict__['node_count']['function']
            methods = self._generate_frame(value=methods,
                                          model=model,
                                          prompt=prompt,
                                          metric="methods")
            
            # Extract the number of convention errors
            convention = results.linter.stats.__dict__['convention']
            convention = self._generate_frame(value=convention,
                                              model=model,
                                              prompt=prompt,
                                              metric="convention")
            
            # Extract the number of errors
            error = results.linter.stats.__dict__['error']
            error = self._generate_frame(value=error,
                                        model=model,
                                        prompt=prompt,
                                        metric="errors")
            
            
            # Extract the number of warnings
            warning = results.linter.stats.__dict__['warning']
            warning = self._generate_frame(value=warning,
                                          model=model,
                                          prompt=prompt,
                                          metric="warnings")
            
            # Extract the number of refactorings
            refactor = results.linter.stats.__dict__['refactor']
            refactor = self._generate_frame(value=refactor,
                                           model=model,
                                           prompt=prompt,
                                           metric="refactor")
            
            # Concatenate the results
            self.df = pd.concat([self.df, score, methods, convention, error, warning, refactor])