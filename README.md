# pdc-final-project-sssp

Do You Need to Make the Graph METIS-Compatible?

Yes, you need to preprocess the graph to ensure:
Vertices are mapped to contiguous 0-based indices.
Duplicate edges are removed.
Self-loops are excluded.
xadj and adjncy are correctly sized.
This preprocessing is critical for real-world graphs, which often have irregular formats.