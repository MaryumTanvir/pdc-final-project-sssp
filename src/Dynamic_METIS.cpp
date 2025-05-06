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
#include <omp.h>
#include <cstdio>
#include <sys/stat.h>
#include <cstdint>

using namespace std;

// Global output file stream for rank 0
ofstream out_file;

// Macro to print to the output file and cerr (rank 0 only)
#define PRINT(rank, msg) do { \
    if (rank == 0) { \
        cerr << msg << endl; \
        if (out_file.is_open()) { \
            out_file << msg << endl; \
        } \
    } \
} while (0)

// Structure to represent an edge
struct Edge
{
    int32_t from, to, weight;
    Edge(int32_t f, int32_t t, int32_t w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{
    int32_t V;                          // Number of vertices
    vector<vector<pair<int32_t, int32_t>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int32_t vertices) : V(vertices), adj(vertices)
    {
        for (auto &list : adj)
            list.reserve(10);
    }
    void addEdge(int32_t u, int32_t v, int32_t w)
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
    void removeEdge(int32_t u, int32_t v)
    {
        auto it_u = find_if(adj[u].begin(), adj[u].end(),
                            [v](const pair<int32_t, int32_t> &e)
                            { return e.first == v; });
        if (it_u != adj[u].end())
            adj[u].erase(it_u);
        auto it_v = find_if(adj[v].begin(), adj[v].end(),
                            [u](const pair<int32_t, int32_t> &e)
                            { return e.first == u; });
        if (it_v != adj[v].end())
            adj[v].erase(it_v);
    }
};

// SSSP tree structure
struct SSSPTree
{
    vector<int32_t> parent;
    vector<int32_t> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int32_t V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false)
    {
        parent.reserve(V);
        dist.reserve(V);
        affected.reserve(V);
        affected_del.reserve(V);
    }
};

// Safe addition to prevent overflow
int32_t safeAdd(int32_t a, int32_t b)
{
    if (a == INT_MAX || b == INT_MAX)
        return INT_MAX;
    if (b > 0 && a > INT_MAX - b)
        return INT_MAX;
    return a + b;
};

// Sequential Dijkstra's algorithm
void sequentialSSSP(const Graph &G, SSSPTree &T, int32_t source)
{
    T.dist[source] = 0;
    priority_queue<pair<int32_t, int32_t>, vector<pair<int32_t, int32_t>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        int32_t d = pq.top().first;
        int32_t u = pq.top().second;
        pq.pop();

        if (d > T.dist[u])
            continue;

        for (const auto &edge : G.adj[u])
        {
            int32_t v = edge.first;
            int32_t w = edge.second;
            int32_t new_dist = safeAdd(T.dist[u], w);
            if (T.dist[v] > new_dist)
            {
                T.dist[v] = new_dist;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

// Computes number of connected components using BFS
int32_t countConnectedComponents(const Graph &G, int rank)
{
    vector<bool> visited(G.V, false);
    int32_t components = 0;

    for (int32_t v = 0; v < G.V; ++v)
    {
        if (!visited[v])
        {
            components++;
            queue<int32_t> q;
            q.push(v);
            visited[v] = true;

            while (!q.empty())
            {
                int32_t u = q.front();
                q.pop();
                for (const auto &edge : G.adj[u])
                {
                    int32_t w = edge.first;
                    if (!visited[w])
                    {
                        visited[w] = true;
                        q.push(w);
                    }
                }
            }
        }
    }

    stringstream ss;
    ss << "Graph has " << components << " connected components";
    PRINT(rank, ss.str());
    return components;
}

// Loads graph from file
bool loadGraph(const string &filename, vector<int32_t> &xadj, vector<int32_t> &adjncy,
               vector<int32_t> &adjwgt, int32_t &nvtxs, int32_t &nedges, Graph &G, int rank)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        stringstream ss;
        ss << "Error: Could not open file: " << filename;
        PRINT(rank, ss.str());
        return false;
    }

    map<int32_t, int32_t> vertex_map;
    set<pair<int32_t, int32_t>> edge_set;
    vector<pair<int32_t, pair<int32_t, int32_t>>> edges;
    edges.reserve(10000000);
    string line;
    int line_count = 0;

    while (getline(file, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        int32_t from, to, weight = 1;
        if (sscanf(line.c_str(), "%d %d %d", &from, &to, &weight) < 2)
        {
            stringstream ss;
            ss << "Error: Invalid format in line " << line_count << ": " << line;
            PRINT(rank, ss.str());
            file.close();
            return false;
        }

        if (from <= 0 || to <= 0)
        {
            stringstream ss;
            ss << "Error: Invalid vertex index in line " << line_count
               << ": from=" << from << ", to=" << to;
            PRINT(rank, ss.str());
            file.close();
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
        if (weight <= 0)
        {
            stringstream ss;
            ss << "Warning: Skipping edge with non-positive weight in line " << line_count
               << ": " << from << " -> " << to << ", weight=" << weight;
            PRINT(rank, ss.str());
            continue;
        }

        vertex_map[from];
        vertex_map[to];
        pair<int32_t, int32_t> edge = {min(from, to), max(from, to)};
        if (edge_set.insert(edge).second)
        {
            edges.emplace_back(from, make_pair(to, weight));
        }
        else
        {
            stringstream ss;
            ss << "Warning: Skipping duplicate edge: " << from << " -> " << to;
            PRINT(rank, ss.str());
        }
    }
    file.close();

    nvtxs = 0;
    for (auto &[orig, new_idx] : vertex_map)
    {
        new_idx = nvtxs++;
    }

    if (nvtxs == 0)
    {
        stringstream ss;
        ss << "Error: No vertices found in graph file";
        PRINT(rank, ss.str());
        return false;
    }

    try
    {
        G = Graph(nvtxs);
        xadj.assign(nvtxs + 1, 0);
        vector<int32_t> degree(nvtxs, 0);

        // Count degrees
        for (const auto &e : edges)
        {
            int32_t u = vertex_map[e.first];
            int32_t v = vertex_map[e.second.first];
            degree[u]++;
            degree[v]++;
        }

        // Build CSR and adjacency list
        xadj[0] = 0;
        for (int32_t i = 0; i < nvtxs; ++i)
        {
            xadj[i + 1] = xadj[i] + degree[i];
        }

        adjncy.resize(xadj[nvtxs]);
        adjwgt.resize(xadj[nvtxs]);
        vector<int32_t> pos(nvtxs, 0);

        for (const auto &e : edges)
        {
            int32_t u = vertex_map[e.first];
            int32_t v = vertex_map[e.second.first];
            int32_t w = e.second.second;
            adjncy[xadj[u] + pos[u]] = v;
            adjwgt[xadj[u] + pos[u]] = w;
            pos[u]++;
            adjncy[xadj[v] + pos[v]] = u;
            adjwgt[xadj[v] + pos[v]] = w;
            pos[v]++;
            G.addEdge(u, v, w);
        }

        edges.clear();
        edge_set.clear();
        vertex_map.clear();
    }
    catch (const std::bad_alloc &e)
    {
        stringstream ss;
        ss << "Error: Memory allocation failed: " << e.what();
        PRINT(rank, ss.str());
        return false;
    }

    nedges = xadj[nvtxs] / 2;

    if (xadj[nvtxs] != static_cast<int32_t>(adjncy.size()))
    {
        stringstream ss;
        ss << "Error: CSR format invalid: xadj[" << nvtxs << "]=" << xadj[nvtxs]
           << ", adjncy.size()=" << adjncy.size();
        PRINT(rank, ss.str());
        return false;
    }

    // Check connectivity
    countConnectedComponents(G, rank);

    stringstream ss;
    ss << "Loaded graph with " << nvtxs << " vertices and " << nedges << " edges";
    PRINT(rank, ss.str());
    return true;
}

// Loads edge changes
vector<pair<pair<int32_t, int32_t>, int32_t>> loadChanges(const string &filename, const Graph &G, int rank)
{
    vector<pair<pair<int32_t, int32_t>, int32_t>> changes;
    if (rank == 0)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            stringstream ss;
            ss << "Error: Could not open changes file: " << filename;
            PRINT(rank, ss.str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        int line_count = 0;
        set<pair<int32_t, int32_t>> change_set;

        while (getline(file, line))
        {
            line_count++;
            if (line.empty() || line[0] == '#')
                continue;

            istringstream iss(line);
            char op;
            int32_t u, v, weight = -1;
            if (!(iss >> op >> u >> v))
            {
                stringstream ss;
                ss << "Error: Invalid format in line " << line_count << ": " << line;
                PRINT(rank, ss.str());
                file.close();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (op == 'I' && !(iss >> weight))
            {
                stringstream ss;
                ss << "Error: Missing weight for insertion in line " << line_count << ": " << line;
                PRINT(rank, ss.str());
                file.close();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            u--; v--;
            if (u < 0 || u >= G.V || v < 0 || v >= G.V || u == v)
            {
                stringstream ss;
                ss << "Error: Invalid vertex index in line " << line_count
                   << ": u=" << u + 1 << ", v=" << v + 1 << ", V=" << G.V;
                PRINT(rank, ss.str());
                file.close();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (op == 'I' && weight <= 0)
            {
                stringstream ss;
                ss << "Error: Non-positive weight for insertion in line " << line_count
                   << ": weight=" << weight;
                PRINT(rank, ss.str());
                file.close();
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            pair<int32_t, int32_t> edge = {min(u, v), max(u, v)};
            if (change_set.find(edge) != change_set.end())
            {
                stringstream ss;
                ss << "Warning: Skipping duplicate change in line " << line_count
                   << ": " << (op == 'I' ? "Insert" : "Delete") << " (" << u + 1 << ", " << v + 1 << ")";
                PRINT(rank, ss.str());
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
                stringstream ss;
                ss << "Warning: Skipping deletion of non-existent edge in line " << line_count
                   << ": (" << u + 1 << ", " << v + 1 << ")";
                PRINT(rank, ss.str());
                continue;
            }
            if (op == 'I' && exists)
            {
                stringstream ss;
                ss << "Warning: Skipping insertion of existing edge in line " << line_count
                   << ": (" << u + 1 << ", " << v + 1 << ")";
                PRINT(rank, ss.str());
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
void partitionGraph(vector<int32_t> &xadj, vector<int32_t> &adjncy,
                    vector<int32_t> &adjwgt, int nparts, vector<int32_t> &parts, int rank)
{
    int32_t nvtxs = xadj.size() - 1;
    if (nvtxs == 0 || adjncy.empty() || adjwgt.empty())
    {
        stringstream ss;
        ss << "Error: Invalid graph data for METIS partitioning: nvtxs=" << nvtxs
           << ", adjncy.size=" << adjncy.size() << ", adjwgt.size=" << adjwgt.size();
        PRINT(rank, ss.str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (nparts > nvtxs)
    {
        stringstream ss;
        ss << "Error: Number of partitions (" << nparts << ") exceeds number of vertices (" << nvtxs << ")";
        PRINT(rank, ss.str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int32_t ncon = 1;
    int32_t objval;
    vector<int32_t> part_idx(nvtxs);
    parts.reserve(nvtxs);
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
    options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_RANDOM;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_NITER] = 10;
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_MINCONN] = 1;
    options[METIS_OPTION_CONTIG] = 0; // Disable contiguous partitioning

    int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL,
                                  adjwgt.data(), &nparts, NULL, NULL, options, &objval, part_idx.data());
    if (ret != METIS_OK)
    {
        stringstream ss;
        ss << "METIS failed with code: " << ret << " for nvtxs=" << nvtxs << ", adjncy.size=" << adjncy.size();
        PRINT(rank, ss.str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    parts.assign(part_idx.begin(), part_idx.end());
    stringstream ss;
    ss << "METIS succeeded. Edge cut: " << objval;
    PRINT(rank, ss.str());
}

// Parallel SSSP update
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int32_t, int32_t>, int32_t>> &changes,
                        const vector<int32_t> &part, int rank, int size)
{
    int32_t V = G.V;
    vector<int32_t> local_vertices;
    local_vertices.reserve(V / size);
    for (int32_t v = 0; v < V; ++v)
    {
        if (part[v] == rank)
            local_vertices.push_back(v);
    }

#pragma omp parallel for schedule(dynamic) num_threads(min(4, omp_get_max_threads()))
    for (size_t i = 0; i < changes.size(); ++i)
    {
        int32_t u = changes[i].first.first;
        int32_t v = changes[i].first.second;
        int32_t w = changes[i].second;

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
            int32_t x = (T.dist[u] > T.dist[v]) ? v : u;
            int32_t y = (T.dist[u] > T.dist[v]) ? u : v;
            int32_t new_dist = safeAdd(T.dist[x], w);
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
    const int max_iterations = min(2 * V, 1000);
    while (global_change && iteration < max_iterations)
    {
        global_change = false;

        bool local_del_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_del_change) num_threads(min(4, omp_get_max_threads()))
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int32_t v = local_vertices[i];
            if (T.affected_del[v])
            {
                T.affected_del[v] = false;
                for (int32_t c = 0; c < V; ++c)
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

#pragma omp parallel for schedule(dynamic) num_threads(min(4, omp_get_max_threads()))
        for (int32_t v = 0; v < V; ++v)
        {
            send_affected_del[v] = T.affected_del[v];
            send_affected[v] = T.affected[v];
        }
        MPI_Allreduce(send_affected_del.data(), recv_affected_del.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(send_affected.data(), recv_affected.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);

#pragma omp parallel for schedule(dynamic) num_threads(min(4, omp_get_max_threads()))
        for (int32_t v = 0; v < V; ++v)
        {
            T.affected_del[v] = recv_affected_del[v];
            T.affected[v] = recv_affected[v];
        }

        bool local_update_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_update_change) num_threads(min(4, omp_get_max_threads()))
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int32_t v = local_vertices[i];
            if (T.affected[v])
            {
                T.affected[v] = false;
                int32_t min_dist = T.dist[v];
                int32_t best_parent = T.parent[v];
                for (const auto &edge : G.adj[v])
                {
                    int32_t n = edge.first;
                    int32_t w = edge.second;
                    int32_t new_dist = safeAdd(T.dist[n], w);
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

        vector<int32_t> send_dist(V, INT_MAX), recv_dist(V);
        vector<int32_t> send_parent(V, -1), recv_parent(V);

#pragma omp parallel for schedule(dynamic) num_threads(min(4, omp_get_max_threads()))
        for (int32_t v = 0; v < V; ++v)
        {
            send_dist[v] = T.dist[v];
            send_parent[v] = T.parent[v];
        }
        MPI_Allreduce(send_dist.data(), recv_dist.data(), V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(send_parent.data(), recv_parent.data(), V, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

#pragma omp parallel for schedule(dynamic) num_threads(min(4, omp_get_max_threads()))
        for (int32_t v = 0; v < V; ++v)
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

    if (rank == 0)
    {
        if (mkdir("../results", 0755) == -1 && errno != EEXIST)
        {
            cerr << "Error: Could not create directory ../results: " << strerror(errno) << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        out_file.open("../results/results.txt");
        if (!out_file.is_open())
        {
            cerr << "Error: Could not open output file: ../results/results.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    string graph_filename = "../data/large_input.txt";
    string changes_filename = "../data/large_update.txt";
    vector<int32_t> xadj, adjncy, adjwgt;
    int32_t nvtxs = 0, nedges = 0;
    Graph G(0);

    try
    {
        if (!loadGraph(graph_filename, xadj, adjncy, adjwgt, nvtxs, nedges, G, rank))
        {
            if (rank == 0 && out_file.is_open())
                out_file.close();
            MPI_Finalize();
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during graph loading: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    vector<int32_t> part(nvtxs);
    try
    {
        partitionGraph(xadj, adjncy, adjwgt, size, part, rank);
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during graph partitioning: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    SSSPTree T(nvtxs);
    if (rank == 0)
    {
        try
        {
            sequentialSSSP(G, T, 0);
        }
        catch (const std::exception &e)
        {
            stringstream ss;
            ss << "Error: Exception during SSSP computation: " << e.what();
            PRINT(rank, ss.str());
            out_file.close();
            MPI_Finalize();
            return -1;
        }
    }

    try
    {
        MPI_Bcast(T.dist.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(T.parent.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during MPI broadcast: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nInitial SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (int32_t i = 0; i < min<int32_t>(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nInitial SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (int32_t i = 0; i < nvtxs; ++i)
            {
                out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                         << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
            }
        }
    }

    auto changes = loadChanges(changes_filename, G, rank);

    int num_changes = changes.size();
    try
    {
        MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during changes broadcast: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    vector<int32_t> change_buf(num_changes * 3);
    if (rank == 0)
    {
        for (int i = 0; i < num_changes; ++i)
        {
            change_buf[i * 3] = changes[i].first.first;
            change_buf[i * 3 + 1] = changes[i].first.second;
            change_buf[i * 3 + 2] = changes[i].second;
        }
    }
    try
    {
        MPI_Bcast(change_buf.data(), num_changes * 3, MPI_INT, 0, MPI_COMM_WORLD);
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during change buffer broadcast: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    changes.clear();
    changes.reserve(num_changes);
    for (int i = 0; i < num_changes; ++i)
    {
        int32_t u = change_buf[i * 3];
        int32_t v = change_buf[i * 3 + 1];
        int32_t w = change_buf[i * 3 + 2];
        changes.push_back({{u, v}, w});
    }

    if (rank == 0)
    {
        stringstream ss;
        ss << "\nApplying " << changes.size() << " edge changes:\n";
        for (const auto &change : changes)
        {
            int32_t u = change.first.first + 1;
            int32_t v = change.first.second + 1;
            int32_t w = change.second;
            ss << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
               << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
        }
        PRINT(rank, ss.str());
    }

    try
    {
        parallelSSSPUpdate(G, T, changes, part, rank, size);
    }
    catch (const std::exception &e)
    {
        stringstream ss;
        ss << "Error: Exception during SSSP update: " << e.what();
        PRINT(rank, ss.str());
        if (rank == 0 && out_file.is_open())
            out_file.close();
        MPI_Finalize();
        return -1;
    }

    if (rank == 0)
    {
        stringstream ss_term;
        ss_term << "\nUpdated SSSP Tree (source vertex 1, first 10 vertices):\n"
                << "Vertex\tDistance\tParent\n";
        for (int32_t i = 0; i < min<int32_t>(10, nvtxs); ++i)
        {
            ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                    << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
        cout << ss_term.str() << endl;

        if (out_file.is_open())
        {
            out_file << "\nUpdated SSSP Tree (source vertex 1, all vertices):\n"
                     << "Vertex\tDistance\tParent\n";
            for (int32_t i = 0; i < nvtxs; ++i)
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