# Application Domain Subfolder Template
Use this folder as a template for to store outputs from LLMs. Structure as follow:

```
.
├── info.yml: Contains prompts used, and other information about application domain, algorithms etc. 
├── <llm_model>.py: Add the provided source code to a new file. 
├── baseline.py: Add the self-produced source code for the problem.
├── documentation/
│   └── <llm_model>.png: Add screenshots during prompting to document the provided responses
```