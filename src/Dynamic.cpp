#include <mpi.h>
#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
#include <climits>
#include <random>
#include <omp.h>

using namespace std;

// Structure to represent an edge
struct Edge
{
    idx_t from, to, weight;
    Edge(idx_t f, idx_t t, idx_t w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{
    int V;                              // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices) : V(vertices), adj(vertices) {}
    // Add edge to the graph (undirected)
    void addEdge(int u, int v, int w)
    {
        if (u >= V || v >= V || u < 0 || v < 0)
        {
            cerr << "Invalid edge: u=" << u << ", v=" << v << ", V=" << V << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
    }
    // Remove edge from the graph (undirected)
    void removeEdge(int u, int v)
    {
        auto it_u = find_if(adj[u].begin(), adj[u].end(),
                            [v](const pair<int, int> &e)
                            { return e.first == v; });
        if (it_u != adj[u].end())
            adj[u].erase(it_u);
        auto it_v = find_if(adj[v].begin(), adj[v].end(),
                            [u](const pair<int, int> &e)
                            { return e.first == u; });
        if (it_v != adj[v].end())
            adj[v].erase(it_v);
    }
};

// SSSP tree structure to store shortest path information
struct SSSPTree
{
    vector<int> parent;
    vector<int> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false) {}
};

// Safe addition to prevent integer overflow in distance calculations
int safeAdd(int a, int b)
{
    if (a == INT_MAX || b == INT_MAX)
        return INT_MAX;
    if (b > 0 && a > INT_MAX - b)
        return INT_MAX;
    return a + b;
};

// Implements Dijkstra's algorithm for initial SSSP tree computation
// (Article: Sequential SSSP Initialization, Section 2, Page 2)
void sequentialSSSP(const Graph &G, SSSPTree &T, int source)
{
    T.dist[source] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (d > T.dist[u])
            continue;

        // Relax edges to neighbors
        for (const auto &edge : G.adj[u])
        {
            int v = edge.first;
            int w = edge.second;
            int new_dist = safeAdd(T.dist[u], w);
            if (T.dist[v] > new_dist)
            {
                T.dist[v] = new_dist;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

// Loads graph from file and constructs adjacency list and METIS CSR format
// (Article: Graph Preprocessing for Dynamic Updates, Section 5, Page 6)
bool loadGraph(const string &filename, vector<idx_t> &xadj, vector<idx_t> &adjncy,
               vector<idx_t> &adjwgt, idx_t &nvtxs, idx_t &nedges, Graph &G, int rank)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        if (rank == 0)
            cerr << "Error: Could not open file: " << filename << endl;
        return false;
    }

    vector<Edge> edges;
    map<idx_t, idx_t> vertex_map;
    set<pair<idx_t, idx_t>> edge_set;
    idx_t max_vertex = 0;
    string line;
    int line_count = 0;

    // Read edge list from file
    while (getline(file, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        istringstream iss(line);
        idx_t from, to;
        idx_t weight = 1;
        if (!(iss >> from >> to))
        {
            if (rank == 0)
                cerr << "Error: Invalid format in line " << line_count << ": " << line << endl;
            return false;
        }
        iss >> weight;

        if (from <= 0 || to <= 0)
        {
            if (rank == 0)
                cerr << "Error: Invalid vertex index in line " << line_count
                     << ": from=" << from << ", to=" << to << endl;
            return false;
        }
        if (from == to)
        {
            if (rank == 0)
                cout << "Warning: Skipping self-loop in line " << line_count
                     << ": " << from << " -> " << to << endl;
            continue;
        }

        vertex_map[from];
        vertex_map[to];
        max_vertex = max(max_vertex, max(from, to));
        edges.emplace_back(from, to, weight);
    }
    file.close();

    // Assign 0-based indices to vertices
    nvtxs = 0;
    for (auto &[orig, new_idx] : vertex_map)
    {
        new_idx = nvtxs++;
    }

    // Initialize graph with adjacency list
    G = Graph(nvtxs);
    vector<Edge> processed_edges;
    for (const auto &e : edges)
    {
        idx_t u = vertex_map[e.from];
        idx_t v = vertex_map[e.to];
        pair<idx_t, idx_t> edge = {min(u, v), max(u, v)};
        if (edge_set.insert(edge).second)
        {
            processed_edges.emplace_back(u, v, e.weight);
            processed_edges.emplace_back(v, u, e.weight);
            G.addEdge(u, v, e.weight);
        }
        else if (rank == 0)
        {
            cout << "Warning: Skipping duplicate edge: " << e.from << " -> " << e.to << endl;
        }
    }

    // Build CSR format for METIS
    sort(processed_edges.begin(), processed_edges.end(),
         [](const Edge &a, const Edge &b)
         { return a.from < b.from; });

    xadj.push_back(0);
    vector<idx_t> degree(nvtxs, 0);
    for (const auto &e : processed_edges)
    {
        degree[e.from]++;
        adjncy.push_back(e.to);
        adjwgt.push_back(e.weight);
    }

    for (idx_t i = 0; i < nvtxs; ++i)
    {
        xadj.push_back(xadj.back() + degree[i]);
    }

    nedges = adjncy.size() / 2;

    if (xadj[nvtxs] != static_cast<idx_t>(adjncy.size()))
    {
        if (rank == 0)
            cerr << "Error: CSR format invalid: xadj[" << nvtxs << "]=" << xadj[nvtxs]
                 << ", adjncy.size()=" << adjncy.size() << endl;
        return false;
    }

    if (rank == 0)
        cout << "Loaded graph with " << nvtxs << " vertices and " << nedges << " edges" << endl;
    return true;
}

// Partitions graph using METIS for load balancing across MPI processes
// (Article: Graph Partitioning for Distributed Parallelism, Section 4, Page 4)
void partitionGraph(vector<idx_t> &xadj, vector<idx_t> &adjncy,
                    vector<idx_t> &adjwgt, int nparts, vector<int> &parts, int rank)
{
    idx_t nvtxs = xadj.size() - 1;
    idx_t ncon = 1;
    idx_t objval;
    vector<idx_t> part_idx(nvtxs);
    // Use METIS to partition graph into nparts, minimizing edge cuts
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL,
                                  adjwgt.data(), &nparts, NULL, NULL, NULL, &objval, part_idx.data());
    if (ret != METIS_OK)
    {
        if (rank == 0)
            cerr << "METIS failed with code: " << ret << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    parts.assign(part_idx.begin(), part_idx.end());
    if (rank == 0)
        cout << "METIS succeeded. Edge cut: " << objval << endl;
}

// Generates random edge changes for testing dynamic updates
// (Article: Dynamic Graph Changes, Section 4, Page 4)
vector<pair<pair<int, int>, int>> generateChanges(const Graph &G, int num_changes, double insert_ratio, int rank)
{
    vector<pair<pair<int, int>, int>> changes;
    if (rank == 0) // Only rank 0 generates changes
    {
        random_device rd;
        mt19937 gen(rd()); // Seed only on rank 0
        uniform_int_distribution<> vertex_dist(0, G.V - 1);
        uniform_int_distribution<> weight_dist(1, 1); // Adjusted weight range for consistency
        bernoulli_distribution insert_dist(insert_ratio);

        // Generate exactly num_changes edge changes
        for (int i = 0; i < num_changes; ++i)
        {
            int u = vertex_dist(gen);
            int v = vertex_dist(gen);
            while (u == v)
                v = vertex_dist(gen);

            bool exists = false;
            for (const auto &edge : G.adj[u])
            {
                if (edge.first == v)
                {
                    exists = true;
                    break;
                }
            }

            if (insert_ratio == 0.0)
            {
                // Only generate deletion edges
                if (exists)
                {
                    changes.push_back({{u, v}, -1}); // Deletion edge
                }
                else
                {
                    --i; // Retry to ensure exactly num_changes deletions
                }
            }
            else
            {
                // Generate both insertions and deletions based on insert_ratio
                int w = insert_dist(gen) ? weight_dist(gen) : (exists ? -1 : weight_dist(gen));
                changes.push_back({{u, v}, w});
            }
        }
    }
    return changes;
}

// Implements parallel dynamic SSSP update algorithm (Article: Algorithm 4 - Asynchronous Update of SSSP, Section 5.1, Page 6)
// Extends shared-memory framework to distributed-memory using MPI and METIS partitioning
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes,
                        const vector<int> &part, int rank, int size)
{
    int V = G.V;
    // Identify local vertices assigned to this MPI process (Article: Load Balancing via Partitioning, Section 4)
    vector<int> local_vertices;
    for (int v = 0; v < V; ++v)
    {
        if (part[v] == rank)
            local_vertices.push_back(v);
    }

// Step 1: Process changed edges in parallel (Article: Algorithm 2 - Identify Affected Vertices, Section 4, Page 4)
// Implements lines 4-20 of Algorithm 2, processing deletions and insertions
#pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i)
    {
        int u = changes[i].first.first;
        int v = changes[i].first.second;
        int w = changes[i].second;

// Update graph structure (protected by critical section for thread safety)
#pragma omp critical
        {
            if (w >= 0)
            { // Insertion (Algorithm 2, lines 11-20)
                G.addEdge(u, v, w);
            }
            else
            { // Deletion (Algorithm 2, lines 4-10)
                G.removeEdge(u, v);
            }
        }

        if (w >= 0)
        { // Handle edge insertion (Algorithm 2, lines 12-19)
            // Determine which vertex may have a shorter path
            int x = (T.dist[u] > T.dist[v]) ? v : u;
            int y = (T.dist[u] > T.dist[v]) ? u : v;
            int new_dist = safeAdd(T.dist[x], w);
            if (T.dist[y] > new_dist)
            {
#pragma omp critical
                {
                    T.dist[y] = new_dist;
                    T.parent[y] = x;
                    T.affected[y] = true; // Mark as affected (Algorithm 2, line 19)
                }
            }
        }
        else
        { // Handle edge deletion (Algorithm 2, lines 5-9)
            // Check if edge is in SSSP tree
            if (T.parent[v] == u)
            {
#pragma omp critical
                {
                    T.dist[v] = INT_MAX; // Set distance to infinity (Algorithm 2, line 7)
                    T.parent[v] = -1;
                    T.affected_del[v] = true; // Mark deletion-affected (Algorithm 2, line 8)
                    T.affected[v] = true;     // Mark as affected (Algorithm 2, line 9)
                }
            }
            else if (T.parent[u] == v)
            {
#pragma omp critical
                {
                    T.dist[u] = INT_MAX;
                    T.parent[u] = -1;
                    T.affected_del[u] = true;
                    T.affected[v] = true;
                }
            }
        }
    }

    // Step 2: Update affected subgraphs iteratively (Article: Algorithm 4, lines 7-end, Section 5.1, Page 6)
    // Combines deletion phase (lines 7-19) and update phase (lines 20-end)
    bool global_change = true;
    while (global_change)
    { // Continue until no changes (Algorithm 4, line 4)
        global_change = false;

        // Deletion Phase: Process deletion-affected vertices (Algorithm 4, lines 7-19)
        // Disconnect children of affected vertices
        bool local_del_change = false;
#pragma omp parallel for reduction(| : local_del_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected_del[v])
            { // Check AffectedDel flag (Algorithm 4, line 8)
                T.affected_del[v] = false;
                // Reset distances for descendants (Algorithm 4, lines 12-15)
                for (int c = 0; c < V; ++c)
                {
                    if (T.parent[c] == v)
                    {
                        T.dist[c] = INT_MAX;
                        T.parent[c] = -1;
                        T.affected_del[c] = true;
                        T.affected[c] = true;
                        local_del_change = true; // Indicate change (Algorithm 4, line 15)
                    }
                }
            }
        }

        // Synchronize affected flags across processes (Article: Distributed Synchronization, Section 4)
        // Replaces queue-based synchronization in Algorithm 4 with MPI
        vector<char> send_affected_del(V, 0), recv_affected_del(V, 0);
        vector<char> send_affected(V, 0), recv_affected(V, 0);
#pragma omp parallel for
        for (int v = 0; v < V; ++v)
        {
            send_affected_del[v] = T.affected_del[v];
            send_affected[v] = T.affected[v];
        }
        MPI_Allreduce(send_affected_del.data(), recv_affected_del.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(send_affected.data(), recv_affected.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
#pragma omp parallel for
        for (int v = 0; v < V; ++v)
        {
            T.affected_del[v] = recv_affected_del[v];
            T.affected[v] = recv_affected[v];
        }

        // Update Phase: Recompute distances for affected vertices (Algorithm 4, lines 20-end)
        bool local_update_change = false;
#pragma omp parallel for reduction(| : local_update_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected[v])
            {                          // Check Affected flag (Algorithm 4, line 25)
                T.affected[v] = false; // Reset flag (Algorithm 4, line 26)
                int min_dist = T.dist[v];
                int best_parent = T.parent[v];
                // Check neighbors for shortest path (Algorithm 4, lines 32-35)
                for (const auto &edge : G.adj[v])
                {
                    int n = edge.first;
                    int w = edge.second;
                    int new_dist = safeAdd(T.dist[n], w);
                    if (new_dist < min_dist)
                    {
                        min_dist = new_dist;
                        best_parent = n;
                        local_update_change = true; // Indicate change (Algorithm 4, line 34)
                    }
                }
                if (min_dist != T.dist[v])
                {
                    T.dist[v] = min_dist;
                    T.parent[v] = best_parent;
                    T.affected[v] = true; // Mark vertex as affected (Algorithm 4, line 37)
                    // Mark neighbors as affected (Algorithm 4, lines 38-41)
                    for (const auto &edge : G.adj[v])
                    {
                        T.affected[edge.first] = true;
                    }
                }
            }
        }

        // Synchronize distances and parents across processes (Article: Ensure Global Consistency, Section 4)
        vector<int> send_dist(V), recv_dist(V);
        vector<int> send_parent(V), recv_parent(V);
#pragma omp parallel for
        for (int v = 0; v < V; ++v)
        {
            send_dist[v] = T.dist[v];
            send_parent[v] = T.parent[v];
        }
        MPI_Allreduce(send_dist.data(), recv_dist.data(), V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(send_parent.data(), recv_parent.data(), V, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#pragma omp parallel for
        for (int v = 0; v < V; ++v)
        {
            T.dist[v] = recv_dist[v];
            T.parent[v] = recv_parent[v];
        }

        // Check for global convergence (Algorithm 4, line 4)
        int local_change_int = local_del_change || local_update_change;
        int global_change_int;
        MPI_Allreduce(&local_change_int, &global_change_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_change = global_change_int;
    }
}

// Main function to orchestrate parallel SSSP computation
int main(int argc, char *argv[])
{
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string filename = "graph.txt";
    vector<idx_t> xadj, adjncy, adjwgt;
    idx_t nvtxs = 0, nedges = 0;
    Graph G(0);

    // Load and preprocess graph (Article: Graph Preprocessing, Section 5, Page 6)
    if (!loadGraph(filename, xadj, adjncy, adjwgt, nvtxs, nedges, G, rank))
    {
        MPI_Finalize();
        return -1;
    }

    // Partition graph across MPI processes (Article: Graph Partitioning, Section 4, Page 4)
    vector<int> part(nvtxs);
    partitionGraph(xadj, adjncy, adjwgt, size, part, rank);

    // Compute initial SSSP tree on rank 0 (Article: Sequential SSSP Initialization, Section 2, Page 2)
    SSSPTree T(nvtxs);
    if (rank == 0)
    {
        sequentialSSSP(G, T, 0);
    }

    // Broadcast initial SSSP tree to all processes (Article: Distributed Initialization, Section 4)
    MPI_Bcast(T.dist.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(T.parent.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);

    // Print initial SSSP tree
    if (rank == 0)
    {
        cout << "\nInitial SSSP Tree (source vertex 1):\n";
        cout << "Vertex\tDistance\tParent\n";
        for (idx_t i = 0; i < nvtxs; ++i)
        {
            cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                 << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    // Generate edge changes for dynamic updates (Article: Dynamic Graph Changes, Section 4, Page 4)
    int num_changes = 5;
    double insert_ratio = 0.0;
    auto changes = generateChanges(G, num_changes, insert_ratio, rank);

    // Gather all changes to all processes (Article: Distributed Change Propagation, Section 4)
    // Broadcast changes from rank 0 to all processes (NEW GROK)
    vector<int> change_buf(num_changes * 3);
    if (rank == 0)
    {
        for (int i = 0; i < num_changes; ++i)
        {
            change_buf[i * 3] = changes[i].first.first;
            change_buf[i * 3 + 1] = changes[i].first.second;
            change_buf[i * 3 + 2] = changes[i].second;
        }
    }
    MPI_Bcast(change_buf.data(), num_changes * 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Reconstruct changes on all processes
    changes.clear();
    for (int i = 0; i < num_changes; ++i)
    {
        int u = change_buf[i * 3];
        int v = change_buf[i * 3 + 1];
        int w = change_buf[i * 3 + 2];
        changes.push_back({{u, v}, w});
    }

    // Print changes
    if (rank == 0)
    {
        cout << "\nApplying " << changes.size() << " edge changes (insert_ratio=" << insert_ratio << "):\n";
        for (const auto &change : changes)
        {
            int u = change.first.first + 1;
            int v = change.first.second + 1;
            int w = change.second;
            cout << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
                 << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
        }
    }

    // Update SSSP tree with dynamic changes (Article: Algorithm 4, Section 5.1, Page 6)
    parallelSSSPUpdate(G, T, changes, part, rank, size);

    // Print updated SSSP tree
    if (rank == 0)
    {
        cout << "\nUpdated SSSP Tree (source vertex 1):\n";
        cout << "Vertex\tDistance\tParent\n";
        for (idx_t i = 0; i < nvtxs; ++i)
        {
            cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                 << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}