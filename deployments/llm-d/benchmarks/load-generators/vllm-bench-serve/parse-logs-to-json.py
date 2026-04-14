import re
import json
import sys

def parse_benchmark_log(file_path):
    results = []
    
    # regex to match: Key name (including units) : Value (float or int)
    # This captures lines like "Mean TTFT (ms): 63.87"
    pattern = re.compile(r"^\s*(.*?):\s+([\d\.]+)\s*$", re.MULTILINE)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Split the file into individual benchmark blocks
        # We split by the starting header
        blocks = content.split("============ Serving Benchmark Result ============")
        
        for block in blocks:
            # Skip empty blocks resulting from the split
            if not block.strip() or "==========" not in block:
                continue
                
            # Find all matches in the current block
            matches = pattern.findall(block)
            
            if matches:
                # Convert the list of tuples into a flattened dictionary
                # We strip extra whitespace from keys
                entry = {key.strip(): float(val) if '.' in val else int(val) 
                         for key, val in matches}
                results.append(entry)
                
        return json.dumps(results, indent=4)

    except FileNotFoundError:
        return json.dumps({"error": "File not found"}, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_log_file>")
    else:
        print(parse_benchmark_log(sys.argv[1]))