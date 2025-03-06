import os
import re

def find_last_max_accuracy(directory):
    accuracy_results = {}
    accuracy_pattern = re.compile(r'Max accuracy: ([0-9]+\.?[0-9]*)%')
    # accuracy_pattern = re.compile(r'Accuracy of the network on the 50000 test images: ([0-9]+\.?[0-9]*)%')

    
    
    for filename in os.listdir(directory):
        if filename.endswith(".log") and "v2_" in filename and "217" in filename:
            filepath = os.path.join(directory, filename)
            last_accuracy = None
            
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    match = accuracy_pattern.search(line)
                    if match:
                        last_accuracy = float(match.group(1))
            
            if last_accuracy is not None:
                accuracy_results[filename] = last_accuracy
    
    return accuracy_results

if __name__ == "__main__":
    log_directory = "./slurm_outs"  # Change this to your log directory
    results = find_last_max_accuracy(log_directory)
    
    for file, accuracy in results.items():
        print(f"File: {file}, Last Max Accuracy: {accuracy}%")
