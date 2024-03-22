import os
import glob 

# Loading script and generating memory_profiler version of the script
class MemoryProfilerScriptGenerator:
    def __init__(self, 
                 model_path:str, 
                 execute_statement:str):
        
        # Path to the Python file
        self.model_path = model_path
        
        # Make sure path ends with '/'
        if self.model_path[-1] != '/':
            self.model_path += '/'
        
        # Get all files in file_path that end with .py
        self.py_files = glob.glob(self.model_path + "*.py")
        
        # Raise error if no files found
        if len(self.py_files) == 0:
            raise FileNotFoundError(f"No .py files found in {self.model_path}")
        
        # Attributes
        self.execute_statement = execute_statement
        
        # Run MemoryProfiler Script Generator
        self.run()
    
    def run(self):
        # Looping through scripts and generating memory profiler version
        for file in self.py_files:
            script = self.load_script(file)
            script_with_memory = self.generate_memory_profiler_script(script=script)
            self.save_memory_profiler_script(script=script_with_memory, file=file)

    def load_script(self, script_path:str):
        with open(script_path, 'r') as f:
            return f.readlines()
    
    def generate_memory_profiler_script(self, script):
        # Memory profiler import statement
        import_statement = """from memory_profiler import profile
        import os
        import psutil
        
        # Get the current process ID
        pid = os.getpid()

        # Create a psutil Process object for the current process
        process = psutil.Process(pid)

        # Get the number of logical CPUs in the system
        num_cores = psutil.cpu_count(logical=True)"""
        
        script = [import_statement] + script
        
        # Adding the decorator to the execute_statement
        script = self.add_decorator_to_functions('@profile', script)
        
        # Assigning the execute_statement
        execute_statement = self.execute_statement
        script = script + [execute_statement]
        
        return script
        
    
    def add_decorator_to_functions(self, decorator, script):
        decorated_lines = []
        for line in script:
            if line.strip().startswith('def'):
                indent_level = len(line) - len(line.lstrip())
                decorated_lines.append(' ' * indent_level + decorator + '\n')
            decorated_lines.append(line)
        return decorated_lines
    
    
    def save_memory_profiler_script(self, script:str, file:str):
        # Folder to use
        path = self.model_path + '/memory_profiler/'
        
        # File name
        name = file.split('/')[-1].replace(".py", "") + "_memory_version.py"
        
        # Generating the folder if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Saving the file
        with open(path + name, 'w') as f:
            f.write(''.join(script))
