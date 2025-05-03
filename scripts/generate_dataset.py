import random

def generate_directed_graph(num_nodes=20, output_file="graph.txt"):
    # Ensure reproducibility
    random.seed(42)
    
    # List to store edges: (from, to, weight)
    edges = []
    
    # Step 1: Create a random directed spanning tree to ensure connectivity
    # Each node (except the first) has exactly one incoming edge
    for to_node in range(2, num_nodes + 1):
        from_node = random.randint(1, to_node - 1)  # Choose a parent from earlier nodes
        weight = 1 if random.random() < 0.8 else random.randint(2, 24)
        edges.append((from_node, to_node, weight))
    
    # Step 2: Add additional random edges for sparsity (e.g., ~30 extra edges)
    num_extra_edges = 30
    for _ in range(num_extra_edges):
        from_node = random.randint(1, num_nodes)
        to_node = random.randint(1, num_nodes)
        while to_node == from_node or (from_node, to_node) in [(e[0], e[1]) for e in edges]:
            to_node = random.randint(1, num_nodes)  # Avoid self-loops and duplicates
        weight = 1 if random.random() < 0.8 else random.randint(2, 24)
        edges.append((from_node, to_node, weight))
    
    # Step 3: Write to file
    with open(output_file, 'w') as f:
        for from_node, to_node, weight in edges:
            f.write(f"{from_node} {to_node} {weight}\n")
    
    print(f"Generated graph with {num_nodes} nodes and {len(edges)} edges in {output_file}")

if __name__ == "__main__":
    generate_directed_graph()