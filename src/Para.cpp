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
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

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
    idx_t from, to, weight;
    Edge(idx_t f, idx_t t, idx_t w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{
    int V;                              // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices, int expected_edges) : V(vertices), adj(vertices)
    {
        // Pre-allocate adjacency lists (avg. degree ~3 + buffer)
        for (auto &list : adj)
            list.reserve(10);
    }
    // Add edge to the graph (undirected)
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
    SSSPTree(int V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false)
    {
        // Pre-allocate to avoid resizing
        parent.reserve(V);
        dist.reserve(V);
        affected.reserve(V);
        affected_del.reserve(V);
    }
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

// Implements parallel Dijkstra's algorithm for initial SSSP tree computation
void parallelSSSP(const Graph &G, SSSPTree &T, int source)
{
    T.dist[source] = 0;
    // Shared priority queue for threads
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    #pragma omp critical
    pq.push({0, source});
    
    vector<bool> visited(G.V, false);
    
    #pragma omp parallel
    {
        // Each thread maintains a local queue to reduce contention
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> local_pq;
        #pragma omp critical
        {
            if (!pq.empty())
            {
                local_pq.push(pq.top());
                pq.pop();
            }
        }
        
        while (true)
        {
            pair<int, int> current;
            bool has_work = false;
            
            // Get a vertex from local queue
            #pragma omp critical
            {
                if (!local_pq.empty())
                {
                    current = local_pq.top();
                    local_pq.pop();
                    has_work = true;
                }
                else if (!pq.empty())
                {
                    current = pq.top();
                    pq.pop();
                    has_work = true;
                }
            }
            
            if (!has_work)
                break;
                
            int d = current.first;
            int u = current.second;
            
            // Skip if already processed or distance is outdated
            if (visited[u] || d > T.dist[u])
                continue;
                
            #pragma omp critical
            visited[u] = true;
            
            // Process neighbors in parallel
            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < G.adj[u].size(); ++i)
            {
                int v = G.adj[u][i].first;
                int w = G.adj[u][i].second;
                int new_dist = safeAdd(T.dist[u], w);
                
                bool updated = false;
                #pragma omp critical
                {
                    if (T.dist[v] > new_dist)
                    {
                        T.dist[v] = new_dist;
                        T.parent[v] = u;
                        updated = true;
                    }
                }
                
                if (updated)
                {
                    #pragma omp critical
                    {
                        local_pq.push({new_dist, v});
                        pq.push({new_dist, v});
                    }
                }
            }
        }
    }
}

// Loads graph from file and constructs adjacency list and METIS CSR format
bool loadGraph(const string &filename, vector<idx_t> &xadj, vector<idx_t> &adjncy,
               vector<idx_t> &adjwgt, idx_t &nvtxs, idx_t &nedges, Graph &G, int rank)
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
    map<idx_t, idx_t> vertex_map;
    set<pair<idx_t, idx_t>> edge_set;
    idx_t max_vertex = 0;
    string line;
    int line_count = 0;

    while (getline(file, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        idx_t from, to;
        idx_t weight = 1;
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
    vector<Edge> processed_edges;
    processed_edges.reserve(12000000);
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
        else
        {
            stringstream ss;
            ss << "Warning: Skipping duplicate edge: " << e.from << " -> " << e.to;
            PRINT(rank, ss.str());
        }
    }

    sort(processed_edges.begin(), processed_edges.end(),
         [](const Edge &a, const Edge &b)
         { return a.from < b.from; });

    xadj.reserve(nvtxs + 1);
    adjncy.reserve(12000000);
    adjwgt.reserve(12000000);
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
        stringstream ss;
        ss << "Error: CSR format invalid: xadj[" << nvtxs << "]=" << xadj[nvtxs]
           << ", adjncy.size()=" << adjncy.size();
        PRINT(rank, ss.str());
        return false;
    }

    stringstream ss;
    ss << "Loaded graph with " << nvtxs << " vertices and " << nedges << " edges";
    PRINT(rank, ss.str());
    return true;
}

// Loads edge changes from a file
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

            u--;
            v--;
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

// Partitions graph using METIS
void partitionGraph(vector<idx_t> &xadj, vector<idx_t> &adjncy,
                    vector<idx_t> &adjwgt, int nparts, vector<int> &parts, int rank)
{
    idx_t nvtxs = xadj.size() - 1;
    idx_t ncon = 1;
    idx_t objval;
    vector<idx_t> part_idx(nvtxs);
    parts.reserve(nvtxs);
    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL,
                                  adjwgt.data(), &nparts, NULL, NULL, NULL, &objval, part_idx.data());
    if (ret != METIS_OK)
    {
        stringstream ss;
        ss << "METIS failed with code: " << ret;
        PRINT(rank, ss.str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    parts.assign(part_idx.begin(), part_idx.end());
    stringstream ss;
    ss << "METIS succeeded. Edge cut: " << objval;
    PRINT(rank, ss.str());
}

// Implements parallel dynamic SSSP update algorithm
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes,
                        const vector<int> &part, int rank, int size)
{
    int V = G.V;
    vector<int> local_vertices;
    local_vertices.reserve(V / size);
    for (int v = 0; v < V; ++v)
    {
        if (part[v] == rank)
            local_vertices.push_back(v);
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); ++i)
    {
        int u = changes[i].first.first;
        int v = changes[i].first.second;
        int w = changes[i].second;

        #pragma omp critical
        {
            if (w >= 0)
            {
                G.addEdge(u, v, w);
            }
            else
            {
                G.removeEdge(u, v);
            }
        }

        if (w >= 0)
        {
            int x = (T.dist[u] > T.dist[v]) ? v : u;
            int y = (T.dist[u] > T.dist[v]) ? u : v;
            int new_dist = safeAdd(T.dist[x], w);
            if (T.dist[y] > new_dist)
            {
                #pragma omp critical
                {
                    T.dist[y] = new_dist;
                    T.parent[y] = x;
                    T.affected[y] = true;
                }
            }
        }
        else
        {
            if (T.parent[v] == u)
            {
                #pragma omp critical
                {
                    T.dist[v] = INT_MAX;
                    T.parent[v] = -1;
                    T.affected_del[v] = true;
                    T.affected[v] = true;
                }
            }
            else if (T.parent[u] == v)
            {
                #pragma omp critical
                {
                    T.dist[u] = INT_MAX;
                    T.parent[u] = -1;
                    T.affected_del[u] = true;
                    T.affected[u] = true;
                }
            }
        }
    }

    bool global_change = true;
    int iteration = 0;
    const int max_iterations = min(2*V, 1000);
    while (global_change && iteration < max_iterations)
    {
        global_change = false;

        bool local_del_change = false;
        #pragma omp parallel for schedule(dynamic) reduction(| : local_del_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected_del[v])
            {
                T.affected_del[v] = false;
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
                for (const auto &edge : G.adj[v])
                {
                    T.affected[edge.first] = true;
                }
            }
        }

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

        int local_change_int = local_del_change || local_update_change;
        int global_change_int;
        MPI_Allreduce(&local_change_int, &global_change_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_change = global_change_int;

        ++iteration;
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "Rank: " << rank << endl;
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
    vector<idx_t> xadj, adjncy, adjwgt;
    idx_t nvtxs = 0, nedges = 0;
    Graph G(0, 0);

    if (!loadGraph(graph_filename, xadj, adjncy, adjwgt, nvtxs, nedges, G, rank))
    {
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    vector<int> part(nvtxs);
    partitionGraph(xadj, adjncy, adjwgt, size, part, rank);

    SSSPTree T(nvtxs);
    if (rank == 0)
    {
        parallelSSSP(G, T, 0);
    }

    MPI_Bcast(T.dist.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(T.parent.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nInitial SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (idx_t i = 0; i < min<idx_t>(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nInitial SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (idx_t i = 0; i < nvtxs; ++i)
            {
                out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                         << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
            }
        }
    }

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

    changes.clear();
    changes.reserve(num_changes);
    for (int i = 0; i < num_changes; ++i)
    {
        int u = change_buf[i * 3];
        int v = change_buf[i * 3 + 1;
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

    parallelSSSPUpdate(G, T, changes, part, rank, size);

    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nUpdated SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (idx_t i = 0; i < min<idx_t>(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nUpdated SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (idx_t i = 0; i < nvtxs; ++i)
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