import glob
import pandas as pd
import radon
from radon.metrics import mi_visit, h_visit
from radon.complexity import cc_visit
from radon import raw



class RadonAnalyzer:
    def __init__(self, model_path:str) -> None:
        """Extracts Radon data for models

        Args:
            model_path (str): Path to model, like 'models/modelname/'
        """
        # Path to model
        self.model_path = model_path
        
        # Make sure path ends with '/'
        if self.model_path[-1] != '/':
            self.model_path += '/'
        
        # Get all files in file_path that end with .py
        self.py_files = glob.glob(self.model_path + "*.py")
        
        # Raise error if no files found
        if len(self.py_files) == 0:
            raise FileNotFoundError(f"No .py files found in {self.model_path}")
        
        # Dataframe for results
        self.df = pd.DataFrame(columns=["metric", "framework", "model", "value"])
        
        # Analyze the code
        self.analyze()

    
    def _load_code(self, filepath: str) -> str:
        """Load code from file

        Args:
            filepath (str): Path to file

        Returns:
            str: Code in file
        """
        # Open file and read code
        with open(filepath, 'r') as file:
            return file.read()
        
    def _generate_frame(self, value, model, prompt, metric):
        # Return the frame
        out = pd.DataFrame({
            "metric": metric,
            "framework": "pylint",
            "model": model,
            "prompt": prompt,
            "value": value}, index=[0])
        
        return out
    
    def analyze(self):
        print("Running radon")
        for file in self.py_files:
            # Load code
            code = self._load_code(file)
            model = file.split("/")[-2]
            prompt = file.split("/")[-1].replace(".py", "")
            print(f"Analyzinng: {model} ({prompt})")
            
            # Analyze code
            print(f"Analyzing {model} ({prompt})")
            try:
                raw_results = self.raw_analyze(code=code, model=model, prompt=prompt)
                cc = self.cc_analyze(code=code, model=model, prompt=prompt)
                hal = self.halstead(code=code, model=model, prompt=prompt)
                mi = self._generate_frame(value=mi_visit(code, multi=True),
                                         model=model,
                                         prompt=prompt,
                                         metric="mi")
            except Exception as e:
                print(f"Error in {model} ({prompt}) with error:\n{e}")
                continue
            
            # Generate frames
            self.df = pd.concat([self.df, raw_results, cc, hal, mi])
            
    
    def raw_analyze(self, code: str, model: str, prompt: str) -> pd.DataFrame:
        """Analyze code

        Args:
            code (str): Code to analyze

        Returns:
            pd.DataFrame: Dataframe with raw data
        """
        # Analyze code
        raw_results = raw.analyze(code)
        
        # Raw Extractor
        loc = self._generate_frame(value=raw_results.loc,
                             model=model,
                             prompt=prompt,
                             metric="loc")

        sloc = self._generate_frame(value=raw_results.sloc,
                             model=model,
                             prompt=prompt,
                             metric="sloc")
        
        comments = self._generate_frame(value=raw_results.comments,
                             model=model,
                             prompt=prompt,
                             metric="comments")

        multi = self._generate_frame(value=raw_results.multi,
                             model=model,
                             prompt=prompt,
                             metric="multi")

        blank = self._generate_frame(value=raw_results.blank,
                             model=model,
                             prompt=prompt,
                             metric="blank")

        # Catching division by zero
        com_to_loc = self._generate_frame(value=raw_results.comments / raw_results.loc,
                             model=model,
                             prompt=prompt,
                             metric="comments_to_loc")
        
        # Catching division by zero
        com_to_sloc = self._generate_frame(value=raw_results.comments / raw_results.sloc,
                             model=model,
                             prompt=prompt,
                             metric="comments_to_sloc")
        # Catching division by zero
        mcom_to_loc = self._generate_frame(value=(raw_results.multi + raw_results.comments) / raw_results.loc,
                             model=model,
                             prompt=prompt,
                             metric="multi_and_comments_to_loc")
        

        return pd.concat([loc, sloc, comments, multi, blank, com_to_loc, com_to_sloc, mcom_to_loc])

    
    def cc_analyze(self, code: str, prompt:str, model:str) -> pd.DataFrame:
        """Analyze code complexity

        Args:
            code (str): Code to analyze

        Returns:
            pd.DataFrame: Dataframe with complexity data
        """
        # Analyze code
        raw = cc_visit(code)
        
        # Complexity Extractor
        is_class: bool = any([x.letter == "C" for x in raw])

        # Take max complexity only from Class element if code contains class
        if is_class:
            for x in raw:
                if x.letter == "C":
                    number_of_methods = len(x.methods)
                    max_complexity = max([y.complexity for y in x.methods])
                else:
                    continue
        
        # Otherwise loop through functions and take max complexity
        else:
            number_of_methods = len(raw)
            max_complexity = max([x.complexity for x in raw])

        mc = self._generate_frame(
            value=max_complexity,
            model=model,
            prompt=prompt,
            metric="max_complexity"
        )
        
        methods = self._generate_frame(
            value=number_of_methods,
            model=model,
            prompt=prompt,
            metric="number_of_methods"
        )
        
        return pd.concat([mc, methods])
    
    def halstead(self, code: str, model:str, prompt: str):
        """Analyze code complexity

        Args:
            code (str): Code to analyze

        Returns:
            dict: Dictionary with complexity data
        """
        halstead = h_visit(code)
        
        # Data Extractor
        length = self._generate_frame(value=halstead.total.length,
                             model=model,
                             prompt=prompt,
                             metric="halstead_length")
        
        volume = self._generate_frame(value=halstead.total.volume,
                             model=model,
                             prompt=prompt,
                             metric="halstead_volume")

        difficulty = self._generate_frame(value=halstead.total.difficulty,
                             model=model,
                             prompt=prompt,
                             metric="halstead_difficulty")

        effort = self._generate_frame(value=halstead.total.effort,
                             model=model,
                             prompt=prompt,
                             metric="halstead_effort")

        time = self._generate_frame(value=halstead.total.time,
                             model=model,
                             prompt=prompt,
                             metric="halstead_time")

        bugs = self._generate_frame(value=halstead.total.bugs,
                             model=model,
                             prompt=prompt,
                             metric="halstead_bugs")
        
        return pd.concat([length, volume, difficulty, effort, time, bugs])
