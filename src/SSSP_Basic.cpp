#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

int main()
{
    string filename = "graph.txt";

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open the file." << endl;
        return -1;
    }

    vector<idx_t> xadj;     // Adjacency index (CSR format)
    vector<idx_t> adjncy;   // Adjacency list (CSR format)
    vector<idx_t> adjwgt;   // Edge weights (optional for METIS)
    vector<idx_t> vertices; // List of all vertices for partitioning

    idx_t nvtxs = 0;  // Number of vertices
    idx_t nedges = 0; // Number of edges
    string line;

    while (getline(file, line))
    {
        istringstream iss(line);
        idx_t from, to, weight;
        if (!(iss >> from >> to >> weight))
        {
            break;
        }

        // Update the number of vertices
        nvtxs = max(nvtxs, max(from, to));
        cout << "Read edge: " << from << " -> " << to << " with weight " << weight << "\n";

        adjncy.push_back(to - 1); // Convert to 0-based index
        adjwgt.push_back(weight);

        adjncy.push_back(from - 1); // Add reverse edge for undirected graph
        adjwgt.push_back(weight);

        nedges++;
    }

    file.close();

    // Construct the xadj array (CSR format)
    xadj.push_back(0); // Starting index for vertex 0
    for (size_t i = 1; i <= nvtxs; ++i)
    {
        xadj.push_back(xadj.back() + 2 * (count(adjncy.begin(), adjncy.end(), i - 1))); // Each vertex has 2 entries for undirected edges
    }

    // METIS partitioning parameters
    idx_t ncon = 1;    // Number of balancing constraints (usually 1 for basic partitioning)
    idx_t nparts = 2;  // Number of partitions
    idx_t objval = 0;  // Edge-cut value
    idx_t part[nvtxs]; // Partition assignments

    // Call METIS to partition the graph
    int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                     NULL, NULL, adjwgt.data(), &nparts,
                                     NULL, NULL, NULL, &objval, part);

    if (status == METIS_OK)
    {
        cout << "METIS succeeded. Edge cut: " << objval << "\nPartitions:\n";
        for (int i = 0; i < nvtxs; ++i)
        {
            cout << "Vertex " << i + 1 << " in part " << part[i] + 1 << "\n"; // Convert to 1-based index for output
        }
    }
    else
    {
        cerr << "METIS failed with status: " << status << "\n";
    }

    return 0;
}
