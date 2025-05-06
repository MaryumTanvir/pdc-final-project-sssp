#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
#include <climits>
#include <omp.h>

using namespace std;

// Global output file stream for rank 0
ofstream out_file;

// Macro to print to the output file (rank 0 only)
#define PRINT(rank, msg) do { \
    if (rank == 0) { \
        if (out_file.is_open()) { \
            out_file << msg << endl; \
        } \
    } \
} while (0)

// Structure to represent an edge
struct Edge
{
    int from, to, weight;
    Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{
    int V;  // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices, int expected_edges) : V(vertices), adj(vertices)
    {
        for (auto &list : adj)
            list.reserve(10);
    }
    void addEdge(int u, int v, int w)
    {
        if (u >= V || v >= V || u < 0 || v < 0)
        {
            stringstream ss;
            ss << "Invalid edge: u=" << u << ", v=" << v << ", V=" << V;
            PRINT(0, ss.str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
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

// SSSP tree structure
struct SSSPTree
{
    vector<int> parent;
    vector<int> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false)
    {
        parent.reserve(V);
        dist.reserve(V);
        affected.reserve(V);
        affected_del.reserve(V);
    }
};

// Safe addition to prevent overflow
int safeAdd(int a, int b)
{
    if (a == INT_MAX || b == INT_MAX)
        return INT_MAX;
    if (b > 0 && a > INT_MAX - b)
        return INT_MAX;
    return a + b;
};

// Sequential Dijkstra's algorithm
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

// Loads graph from file
bool loadGraph(const string &filename, int &nvtxs, int &nedges, Graph &G, int rank)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        stringstream ss;
        ss << "Error: Could not open file: " << filename;
        PRINT(rank, ss.str());
        return false;
    }

    vector<Edge> edges;
    edges.reserve(6000000);
    map<int, int> vertex_map;
    set<pair<int, int>> edge_set;
    int max_vertex = 0;
    string line;
    int line_count = 0;

    while (getline(file, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        int from, to, weight = 1;
        if (sscanf(line.c_str(), "%d %d %d", &from, &to, &weight) < 2)
        {
            stringstream ss;
            ss << "Error: Invalid format in line " << line_count << ": " << line;
            PRINT(rank, ss.str());
            return false;
        }

        if (from <= 0 || to <= 0)
        {
            stringstream ss;
            ss << "Error: Invalid vertex index in line " << line_count
               << ": from=" << from << ", to=" << to;
            PRINT(rank, ss.str());
            return false;
        }
        if (from == to)
        {
            stringstream ss;
            ss << "Warning: Skipping self-loop in line " << line_count
               << ": " << from << " -> " << to;
            PRINT(rank, ss.str());
            continue;
        }

        vertex_map[from];
        vertex_map[to];
        max_vertex = max(max_vertex, max(from, to));
        edges.emplace_back(from, to, weight);
    }
    file.close();

    nvtxs = 0;
    for (auto &[orig, new_idx] : vertex_map)
    {
        new_idx = nvtxs++;
    }

    G = Graph(nvtxs, 6000000);
    nedges = 0;
    for (const auto &e : edges)
    {
        int u = vertex_map[e.from];
        int v = vertex_map[e.to];
        pair<int, int> edge = {min(u, v), max(u, v)};
        if (edge_set.insert(edge).second)
        {
            G.addEdge(u, v, e.weight);
            nedges++;
        }
        else
        {
            stringstream ss;
            ss << "Warning: Skipping duplicate edge: " << e.from << " -> " << e.to;
            PRINT(rank, ss.str());
        }
    }

    stringstream ss;
    ss << "Loaded graph with " << nvtxs << " vertices and " << nedges << " edges";
    PRINT(rank, ss.str());
    return true;
}

// Loads edge changes
vector<pair<pair<int, int>, int>> loadChanges(const string &filename, const Graph &G, int rank)
{
    vector<pair<pair<int, int>, int>> changes;
    if (rank == 0)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: Could not open changes file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        int line_count = 0;
        set<pair<int, int>> change_set;

        while (getline(file, line))
        {
            line_count++;
            if (line.empty() || line[0] == '#')
                continue;

            istringstream iss(line);
            char op;
            int u, v, weight = -1;
            if (!(iss >> op >> u >> v))
            {
                cerr << "Error: Invalid format in line " << line_count << ": " << line << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (op == 'I' && !(iss >> weight))
            {
                cerr << "Error: Missing weight for insertion in line " << line_count << ": " << line << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            u--; v--;
            if (u < 0 || u >= G.V || v < 0 || v >= G.V || u == v)
            {
                cerr << "Error: Invalid vertex index in line " << line_count
                     << ": u=" << u + 1 << ", v=" << v + 1 << ", V=" << G.V << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            pair<int, int> edge = {min(u, v), max(u, v)};
            if (change_set.find(edge) != change_set.end())
            {
                cout << "Warning: Skipping duplicate change in line " << line_count
                     << ": " << (op == 'I' ? "Insert" : "Delete") << " (" << u + 1 << ", " << v + 1 << ")" << endl;
                continue;
            }

            bool exists = false;
            for (const auto &e : G.adj[u])
            {
                if (e.first == v)
                {
                    exists = true;
                    break;
                }
            }

            if (op == 'D' && !exists)
            {
                cout << "Warning: Skipping deletion of non-existent edge in line " << line_count
                     << ": (" << u + 1 << ", " << v + 1 << ")" << endl;
                continue;
            }
            if (op == 'I' && exists)
            {
                cout << "Warning: Skipping insertion of existing edge in line " << line_count
                     << ": (" << u + 1 << ", " << v + 1 << ")" << endl;
                continue;
            }

            changes.push_back({{u, v}, (op == 'I' ? weight : -1)});
            change_set.insert(edge);
        }
        file.close();
    }
    return changes;
}

// Implements parallel dynamic SSSP update algorithm
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes,
                        int rank, int size)
{
    // Algorithm 2 - Identifying Affected Vertices
    int V = G.V;
    vector<int> local_vertices;
    local_vertices.reserve(V / size);
    for (int v = rank; v < V; v += size)
    {
        local_vertices.push_back(v);
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); ++i)
    {
        int u = changes[i].first.first;
        int v = changes[i].first.second;
        int w = changes[i].second;

        if (w >= 0)
        {
            G.addEdge(u, v, w);
        }
        else
        {
            G.removeEdge(u, v);
        }

        if (w >= 0)
        {
            int x = (T.dist[u] > T.dist[v]) ? v : u;
            int y = (T.dist[u] > T.dist[v]) ? u : v;
            int new_dist = safeAdd(T.dist[x], w);
            if (T.dist[y] > new_dist)
            {
                T.dist[y] = new_dist;
                T.parent[y] = x;
                T.affected[y] = true;
            }
        }
        else
        {
            if (T.parent[v] == u)
            {
                T.dist[v] = INT_MAX;
                T.parent[v] = -1;
                T.affected_del[v] = true;
                T.affected[v] = true;
            }
            else if (T.parent[u] == v)
            {
                T.dist[u] = INT_MAX;
                T.parent[u] = -1;
                T.affected_del[u] = true;
                T.affected[u] = true; // Mark the child vertex as affected
            }
        }
    }

    // Algorithm 4 - Updating Affected Vertices
    bool global_change = true;
    int iteration = 0;
    const int max_iterations = min(2 * V, 1000);
    while (global_change && iteration < max_iterations)
    {
        global_change = false;

        // Deletion Phase: Process deletion-affected vertices
        bool local_del_change = false;
        #pragma omp parallel for schedule(dynamic) reduction(| : local_del_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected_del[v])
            {
                T.affected_del[v] = false;
                // Reset distances for descendants
                for (int c = 0; c < V; ++c)
                {
                    if (T.parent[c] == v)
                    {
                        T.dist[c] = INT_MAX;
                        T.parent[c] = -1;
                        T.affected_del[c] = true;
                        T.affected[c] = true;
                        local_del_change = true;
                    }
                }
                // Mark neighbors as affected to process them as well
                for (const auto &edge : G.adj[v])
                {
                    T.affected[edge.first] = true;
                }
            }
        }

        // Synchronize affected flags across processes
        vector<char> send_affected_del(V, 0), recv_affected_del(V, 0);
        vector<char> send_affected(V, 0), recv_affected(V, 0);

        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < V; ++v)
        {
            send_affected_del[v] = T.affected_del[v];
            send_affected[v] = T.affected[v];
        }
        MPI_Allreduce(send_affected_del.data(), recv_affected_del.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(send_affected.data(), recv_affected.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < V; ++v)
        {
            T.affected_del[v] = recv_affected_del[v];
            T.affected[v] = recv_affected[v];
        }

        // Update Phase: Recompute distances for affected vertices
        bool local_update_change = false;
        #pragma omp parallel for schedule(dynamic) reduction(| : local_update_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected[v])
            {
                T.affected[v] = false;
                int min_dist = T.dist[v];
                int best_parent = T.parent[v];
                for (const auto &edge : G.adj[v])
                {
                    int n = edge.first;
                    int w = edge.second;
                    int new_dist = safeAdd(T.dist[n], w);
                    if (new_dist < min_dist)
                    {
                        min_dist = new_dist;
                        best_parent = n;
                        local_update_change = true;
                    }
                }
                if (min_dist != T.dist[v])
                {
                    T.dist[v] = min_dist;
                    T.parent[v] = best_parent;
                    T.affected[v] = true;
                    for (const auto &edge : G.adj[v])
                    {
                        T.affected[edge.first] = true;
                    }
                }
            }
        }

        vector<int> send_dist(V), recv_dist(V);
        vector<int> send_parent(V), recv_parent(V);

        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < V; ++v)
        {
            send_dist[v] = T.dist[v];
            send_parent[v] = T.parent[v];
        }
        MPI_Allreduce(send_dist.data(), recv_dist.data(), V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(send_parent.data(), recv_parent.data(), V, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(dynamic)
        for (int v = 0; v < V; ++v)
        {
            T.dist[v] = recv_dist[v];
            T.parent[v] = recv_parent[v];
        }

        // Check for global convergence
        int local_change_int = local_del_change || local_update_change;
        int global_change_int;
        MPI_Allreduce(&local_change_int, &global_change_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_change = global_change_int;

        ++iteration;

        /*
        if (rank == 0) {
            cout << "Rank " << rank << ": Iteration " << iteration << ", global_change=" << global_change << endl;
            for (int v = 0; v < V; ++v) {
                if (T.affected[v]) {
                    cout << "Rank " << rank << ": Vertex " << v + 1 << " is affected" << endl;
                }
            }
        }
        */
    }

    if (iteration >= max_iterations && rank == 0)
    {
        stringstream ss;
        ss << "Warning: Reached maximum iterations (" << max_iterations << "). Possible non-convergence.";
        PRINT(rank, ss.str());
    }
}

// Main function
int main(int argc, char *argv[])
{
    // MPI Environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Rank: " << rank << endl;
    // open output file
    if (rank == 0)
    {
        out_file.open("../results/results.txt");
        if (!out_file.is_open())
        {
            cout << "Error: Could not open output file: ../results/results.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    string graph_filename = "../data/graph.txt";
    string changes_filename = "../data/update_small.txt";
    int nvtxs = 0, nedges = 0;
    Graph G(0, 0);

    //Load and preprocess graph
    if (!loadGraph(graph_filename, nvtxs, nedges, G, rank))
    {
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    // Initial SSSP Tree
    SSSPTree T(nvtxs);
    if (rank == 0)
    {
        sequentialSSSP(G, T, 0);
    }

    // Broadcast initial tree to every process
    MPI_Bcast(T.dist.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(T.parent.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);

    // Output initial SSSP tree
    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nInitial SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < min(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nInitial SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (int i = 0; i < nvtxs; ++i)
            {
                out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                         << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
            }
        }
    }

    // Load updates from file & broadcast to all processes
    auto changes = loadChanges(changes_filename, G, rank);

    int num_changes = changes.size();
    MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    changes.reserve(num_changes);
    for (int i = 0; i < num_changes; ++i)
    {
        int u = change_buf[i * 3];
        int v = change_buf[i * 3 + 1];
        int w = change_buf[i * 3 + 2];
        changes.push_back({{u, v}, w});
    }

    if (rank == 0)
    {
        stringstream ss;
        ss << "\nApplying " << changes.size() << " edge changes:\n";
        for (const auto &change : changes)
        {
            int u = change.first.first + 1;
            int v = change.first.second + 1;
            int w = change.second;
            ss << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
               << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
        }
        PRINT(rank, ss.str());
    }

    // ALGORITHM - Update SSSP Tree
    parallelSSSPUpdate(G, T, changes, rank, size);

    // Output updated SSSP tree
    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nUpdated SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < min(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nUpdated SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (int i = 0; i < nvtxs; ++i)
            {
                out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                         << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
            }
        }
    }

    if (rank == 0 && out_file.is_open())
        out_file.close();
    
    MPI_Finalize();
    return 0;
}