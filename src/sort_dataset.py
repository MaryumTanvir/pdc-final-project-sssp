import sys

def sort_output_file(input_file, output_file):
    # Read the lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse each line into a tuple of (from_vertex, to_vertex, weight)
    edges = []
    for line in lines:
        try:
            from_v, to_v, weight = map(int, line.strip().split())
            edges.append((from_v, to_v, weight))
        except ValueError:
            print(f"Skipping invalid line: {line.strip()}")
    
    # Sort edges based on from_vertex
    edges.sort(key=lambda x: x[0])
    
    # Write sorted edges to the output file
    with open(output_file, 'w') as f:
        for from_v, to_v, weight in edges:
            f.write(f"{from_v} {to_v} {weight}\n")

if __name__ == "__main__":
    
    input_file = "large_input.txt"
    output_file = "sort_output.txt"
    sort_output_file(input_file, output_file)
    print(f"Sorted output written to {output_file}")