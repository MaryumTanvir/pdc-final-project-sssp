import random

def generate_updates(output_file, num_updates=100, max_vertex=20, max_weight=100):
    updates = []
    # Generate 50 insertions and 50 deletions
    for _ in range(num_updates // 2):
        # Insertion: I from to weight
        from_v = random.randint(1, max_vertex)
        to_v = random.randint(1, max_vertex)
        while to_v == from_v:  # Avoid self-loops
            to_v = random.randint(1, max_vertex)
        weight = random.randint(1, max_weight)
        updates.append(f"I {from_v} {to_v} {weight}")
        
        # Deletion: D from to
        from_v = random.randint(1, max_vertex)
        to_v = random.randint(1, max_vertex)
        while to_v == from_v:
            to_v = random.randint(1, max_vertex)
        updates.append(f"D {from_v} {to_v}")
    
    # Shuffle updates to mix insertions and deletions
    random.shuffle(updates)
    
    # Write to file
    with open(output_file, 'w') as f:
        for update in updates:
            f.write(f"{update}\n")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    generate_updates("update_small.txt")
    print("Generated update_small with 100 edge updates")