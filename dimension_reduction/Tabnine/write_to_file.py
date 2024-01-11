import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-E', '--eval', type=str, default='pylint')
parser.add_argument('-F', '--file', type=str, default='all')
parser.add_argument('-O', '--overwrite', type=bool, default=False)

args = parser.parse_args()

writer = '>>' if not args.overwrite else '>'
model = os.getcwd().split('/')[-1]

try:
    file_out = model + '_' + args.eval + '_' + args.file.rstrip('.py') 
except:
    file_out = model + '_' + args.eval

if not os.path.exists('evaluation'):
    os.makedirs('evaluation')

if args.eval == 'pylint':
    if args.file == 'all':
        fileout = f'evaluation/{model}_pylint'
        for file in ['long.py', 'medium.py', 'small.py']:
            writer = '>' if file == 'long.py' else '>>'
            print('Linting: ', file)
            os.system(f'echo "\n----{file}----\n" {writer} {fileout}.txt')
            os.system(f'pylint {file} >> {fileout}.txt')
        print('Finished')
    else:
        os.system(f'pylint {args.file} {writer} {file_out}.txt')
        

elif args.eval == "radon":
    if args.file == 'all':
        filename_out = f'evaluation/{model}_radon'
        
        for file in ['long.py', 'medium.py','small.py']:
            writer = '>' if file == 'long.py' else '>>'
            file_title = file.strip('.py').upper()
            os.system(f'echo "\n----{file_title}----\n" {writer} {filename_out}.txt')
            
            for radon_eval in ['mi', 'cc', 'raw', 'hal']:
                if radon_eval == 'cc':
                    write = "Cyclomatic Complexity (CC)"
                    extra_argument = '-a'
                elif radon_eval == 'mi':
                    write = "\nMaintainability Index"
                    extra_argument = '-s'
                elif radon_eval == 'raw':
                    write = "\nRaw"
                    extra_argument = ''
                else:
                    write = "\nHalstead Metric"
                    extra_argument = ''

                os.system(f'echo "{write}" >> {filename_out}.txt')
                os.system(f'radon {radon_eval} {file} {extra_argument} >> {filename_out}.txt')

elif args.eval == "flake8":
    flake8_file_out = f'evaluation/{model}_flake8'
    for file in ['long.py', 'medium.py','small.py']:
        writer = '>' if file == 'long.py' else '>>'
        os.system(f'flake8 {file} {writer} {flake8_file_out}_raw.txt')
    os.system(f'python flake8_summarizer.py {flake8_file_out}_raw.txt > {flake8_file_out}_processed.txt')
    
elif args.eval == "memory_usage": 
    fileout = f'evaluation/{model}_memory_usage'
    
    for file in ['long', 'medium','small']:
        print(f'Computing memory usage of: {file}')
        writer = '>' if file == 'long' else '>>'
        os.system(f'echo "\n----{file}----\n" {writer} {fileout}.txt')
        os.system(f'python -m memory_profiler memory_usage/{file}_memory_usage_version.py >> {fileout}.txt')