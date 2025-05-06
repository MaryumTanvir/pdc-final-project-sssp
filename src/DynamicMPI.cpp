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
#include <mpi.h>

using namespace std;

// Structure to represent an edge (Section 2, Page 2: Graph representation)
struct Edge
{
    int from, to, weight;
    Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list (Section 2, Page 2: Graph representation)
struct Graph
{
    int V;                              // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices) : V(vertices), adj(vertices) {}
    // Add edge to the graph (undirected) (Section 4, Page 4: Edge Insertion)
    void addEdge(int u, int v, int w)
    {
        if (u >= V || v >= V || u < 0 || v < 0)
        {
            cerr << "Invalid edge: u=" << u << ", v=" << v << ", V=" << V << endl;
            exit(1);
        }
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
    }
    // Remove edge from the graph (undirected) (Section 4, Page 4: Edge Deletion)
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

// SSSP tree structure to store shortest path information (Section 2, Page 2: SSSP Tree)
struct SSSPTree
{
    vector<int> parent;
    vector<int> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false) {}
};

// Safe addition to prevent integer overflow in distance calculations (Section 4, Page 4: Safe Distance Updates)
int safeAdd(int a, int b)
{
    if (a == INT_MAX || b == INT_MAX)
        return INT_MAX;
    if (b > 0 && a > INT_MAX - b)
        return INT_MAX;
    return a + b;
};

// Implements Dijkstra's algorithm for initial SSSP tree computation
// (Section 2, Page 2: Sequential SSSP Initialization)
// Run only on rank 0, then broadcast results
void sequentialSSSP(const Graph &G, SSSPTree &T, int source, int rank)
{
    if (rank == 0)
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

    // Broadcast distances and parents to all processes
    MPI_Bcast(T.dist.data(), T.dist.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(T.parent.data(), T.parent.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

// Algorithm 1: Updates SSSP for a single edge change (Section 2.1, Page 2: Sequential SSSP Update)
void singleChange(Graph &G, SSSPTree &T, const pair<pair<int, int>, int> &change)
{
    int u = change.first.first;
    int v = change.first.second;
    int w = change.second;
    int x, y;

    // Find the affected vertex (Algorithm 1, lines 3-7)
    if (T.dist[u] > T.dist[v])
    {
        x = v;
        y = u;
    }
    else
    {
        x = u;
        y = v;
    }

    // Update graph and SSSP tree (Algorithm 1, lines 10-13)
    if (w >= 0)
    {
        G.addEdge(u, v, w);
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
        G.removeEdge(u, v);
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
            T.affected[v] = true;
        }
    }

    // Update affected subgraph (Algorithm 1, lines 15-23)
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    if (T.affected[y])
        pq.push({T.dist[y], y});

    while (!pq.empty())
    {
        int z = pq.top().second;
        pq.pop();
        T.affected[z] = false;

        int min_dist = T.dist[z];
        int best_parent = T.parent[z];
        bool updated = false;

        for (const auto &edge : G.adj[z])
        {
            int n = edge.first;
            int weight = edge.second;
            int new_dist = safeAdd(T.dist[n], weight);
            if (new_dist < min_dist)
            {
                min_dist = new_dist;
                best_parent = n;
                updated = true;
            }
        }

        if (updated)
        {
            T.dist[z] = min_dist;
            T.parent[z] = best_parent;
            T.affected[z] = true;
            for (const auto &edge : G.adj[z])
            {
                if (!T.affected[edge.first])
                {
                    T.affected[edge.first] = true;
                    pq.push({T.dist[edge.first], edge.first});
                }
            }
        }
    }
}

// Loads graph from file and constructs adjacency list
// (Section 5, Page 6: Graph Preprocessing for Dynamic Updates)
bool loadGraph(const string &filename, Graph &G, int rank)
{
    if (rank == 0)
    {
        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: Could not open file: " << filename << endl;
            return false;
        }

        vector<Edge> edges;
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

            istringstream iss(line);
            int from, to;
            int weight = 1;
            if (!(iss >> from >> to))
            {
                cerr << "Error: Invalid format in line " << line_count << ": " << line << endl;
                return false;
            }
            iss >> weight;

            if (from <= 0 || to <= 0)
            {
                cerr << "Error: Invalid vertex index in line " << line_count
                     << ": from=" << from << ", to=" << to << endl;
                return false;
            }
            if (from == to)
            {
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

        int nvtxs = 0;
        for (auto &[orig, new_idx] : vertex_map)
        {
            new_idx = nvtxs++;
        }

        G = Graph(nvtxs);
        for (const auto &e : edges)
        {
            int u = vertex_map[e.from];
            int v = vertex_map[e.to];
            pair<int, int> edge = {min(u, v), max(u, v)};
            if (edge_set.insert(edge).second)
            {
                G.addEdge(u, v, e.weight);
            }
        }

        cout << "Loaded graph with " << nvtxs << " vertices and " << edge_set.size() << " edges" << endl;
    }

    // Broadcast graph size
    int V = G.V;
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        G = Graph(V);

    // Serialize and broadcast adjacency list
    vector<int> send_buf;
    if (rank == 0)
    {
        for (int u = 0; u < G.V; ++u)
        {
            send_buf.push_back(G.adj[u].size());
            for (const auto &e : G.adj[u])
            {
                send_buf.push_back(e.first);
                send_buf.push_back(e.second);
            }
        }
    }

    int send_size = send_buf.size();
    MPI_Bcast(&send_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
        send_buf.resize(send_size);
    MPI_Bcast(send_buf.data(), send_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        int pos = 0;
        for (int u = 0; u < G.V; ++u)
        {
            int num_edges = send_buf[pos++];
            for (int i = 0; i < num_edges; ++i)
            {
                int v = send_buf[pos++];
                int w = send_buf[pos++];
                G.adj[u].push_back({v, w});
            }
        }
    }

    return true;
}

// Generates random edge changes for testing dynamic updates
// (Section 4, Page 4: Dynamic Graph Changes)
vector<pair<pair<int, int>, int>> generateChanges(const Graph &G, int num_changes, double insert_ratio, int rank)
{
    vector<pair<pair<int, int>, int>> changes;
    if (rank == 0)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> vertex_dist(0, G.V - 1);
        uniform_int_distribution<> weight_dist(1, 1);
        bernoulli_distribution insert_dist(insert_ratio);

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
                if (exists)
                {
                    changes.push_back({{u, v}, -1});
                }
                else
                {
                    --i;
                }
            }
            else
            {
                int w = insert_dist(gen) ? weight_dist(gen) : (exists ? -1 : weight_dist(gen));
                changes.push_back({{u, v}, w});
            }
        }
    }

    // Broadcast changes
    int change_size = changes.size();
    MPI_Bcast(&change_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> change_buf;
    if (rank == 0)
    {
        for (const auto &c : changes)
        {
            change_buf.push_back(c.first.first);
            change_buf.push_back(c.first.second);
            change_buf.push_back(c.second);
        }
    }
    else
    {
        change_buf.resize(change_size * 3);
    }
    MPI_Bcast(change_buf.data(), change_size * 3, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        changes.clear();
        for (int i = 0; i < change_size; ++i)
        {
            int u = change_buf[i * 3];
            int v = change_buf[i * 3 + 1];
            int w = change_buf[i * 3 + 2];
            changes.push_back({{u, v}, w});
        }
    }

    return changes;
}

// Determines which process owns a vertex (block partitioning)
int getOwner(int vertex, int V, int num_processes)
{
    return (vertex * num_processes) / V;
}

// Implements parallel dynamic SSSP update algorithm (Algorithm 4, Section 5.1, Page 6)
// Uses hybrid MPI+OpenMP for distributed and shared-memory parallelism
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes, int async_level, int batch_size, int rank, int num_processes)
{
    int V = G.V;
    vector<pair<pair<int, int>, int>> deletions, insertions;

    // Separate deletions and insertions (Section 5.1, Page 6: Process deletions before insertions)
    for (const auto &change : changes)
    {
        if (change.second < 0)
            deletions.push_back(change);
        else
            insertions.push_back(change);
    }

    // Compute vertex ownership for block partitioning
    vector<int> local_vertices;
    for (int v = 0; v < V; ++v)
    {
        if (getOwner(v, V, num_processes) == rank)
            local_vertices.push_back(v);
    }

    // Process changes in batches (Section 5.1, Page 7: Processing Batches of Changes)
    auto processBatch = [&](const vector<pair<pair<int, int>, int>> &batch, bool is_deletion)
    {
    // Step 1: Identify affected vertices (Algorithm 2, Section 4, Page 4)
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < batch.size(); ++i)
        {
            int u = batch[i].first.first;
            int v = batch[i].first.second;
            int w = batch[i].second;

#pragma omp critical
            {
                if (w >= 0)
                {
                    G.addEdge(u, v, w); // Algorithm 2, line 20
                }
                else
                {
                    G.removeEdge(u, v); // Algorithm 2, line 10
                }
            }

            if (w >= 0)
            {
                // Edge insertion (Algorithm 2, lines 12-19)
                int x = (T.dist[u] > T.dist[v]) ? v : u;
                int y = (T.dist[u] > T.dist[v]) ? u : v;
                int new_dist = safeAdd(T.dist[x], w);
                if (T.dist[y] > new_dist)
                {
#pragma omp critical
                    {
                        T.dist[y] = new_dist;
                        T.parent[y] = x;
                        T.affected[y] = true; // Algorithm 2, line 19
                    }
                }
            }
            else
            {
                // Edge deletion (Algorithm 2, lines 5-9)
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
                        T.affected[v] = true;
                    }
                }
            }
        }
    };

    // Process deletions in batches
    for (size_t i = 0; i < deletions.size(); i += batch_size)
    {
        vector<pair<pair<int, int>, int>> batch(
            deletions.begin() + i,
            deletions.begin() + min(i + batch_size, deletions.size()));
        processBatch(batch, true);
    }

    // Process insertions in batches
    for (size_t i = 0; i < insertions.size(); i += batch_size)
    {
        vector<pair<pair<int, int>, int>> batch(
            insertions.begin() + i,
            insertions.begin() + min(i + batch_size, insertions.size()));
        processBatch(batch, false);
    }

    // Step 2: Update affected subgraphs (Algorithm 4, Section 5.1, Page 6)
    bool global_change = true;
    while (global_change)
    {
        global_change = false;

        // Synchronize affected flags across processes
        vector<char> send_affected(V), recv_affected(V);
        vector<char> send_affected_del(V), recv_affected_del(V);
        for (int v = 0; v < V; ++v)
        {
            send_affected[v] = T.affected[v] ? 1 : 0;
            send_affected_del[v] = T.affected_del[v] ? 1 : 0;
        }
        MPI_Allreduce(send_affected.data(), recv_affected.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(send_affected_del.data(), recv_affected_del.data(), V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        for (int v = 0; v < V; ++v)
        {
            T.affected[v] = recv_affected[v];
            T.affected_del[v] = recv_affected_del[v];
        }

        // Deletion Phase (Algorithm 4, lines 7-19)
        bool local_del_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_del_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected_del[v])
            {
                T.affected_del[v] = false;
                queue<int> Q;
                Q.push(v);
                int level = 0;

                while (!Q.empty() && level <= async_level)
                {
                    int x = Q.front();
                    Q.pop();

                    for (int c = 0; c < V; ++c)
                    {
                        if (T.parent[c] == x)
                        {
                            T.dist[c] = INT_MAX;
                            T.parent[c] = -1;
                            T.affected_del[c] = true;
                            T.affected[c] = true;
                            local_del_change = true;
                            if (level < async_level)
                                Q.push(c);
                        }
                    }
                    level++;
                }

                for (const auto &edge : G.adj[v])
                {
                    T.affected[edge.first] = true;
                }
            }
        }

        // Update Phase (Algorithm 4, lines 20-end)
        bool local_update_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_update_change)
        for (size_t i = 0; i < local_vertices.size(); ++i)
        {
            int v = local_vertices[i];
            if (T.affected[v])
            {
                T.affected[v] = false;
                queue<int> Q;
                Q.push(v);
                int level = 0;

                while (!Q.empty() && level <= async_level)
                {
                    int x = Q.front();
                    Q.pop();

                    int min_dist = T.dist[x];
                    int best_parent = T.parent[x];
                    bool updated = false;

                    for (const auto &edge : G.adj[x])
                    {
                        int n = edge.first;
                        int w = edge.second;
                        int new_dist = safeAdd(T.dist[n], w);
                        if (new_dist < min_dist)
                        {
                            min_dist = new_dist;
                            best_parent = n;
                            updated = true;
                        }
                    }

                    if (updated)
                    {
                        T.dist[x] = min_dist;
                        T.parent[x] = best_parent;
                        T.affected[x] = true;
                        local_update_change = true;

                        for (const auto &edge : G.adj[x])
                        {
                            T.affected[edge.first] = true;
                        }

                        if (level < async_level)
                            Q.push(x);
                    }

                    level++;
                }
            }
        }

        // Synchronize distances and parents for boundary vertices
        vector<int> send_dist(V), recv_dist(V);
        vector<int> send_parent(V), recv_parent(V);
        for (int v = 0; v < V; ++v)
        {
            send_dist[v] = T.dist[v];
            send_parent[v] = T.parent[v];
        }
        MPI_Allreduce(send_dist.data(), recv_dist.data(), V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(send_parent.data(), recv_parent.data(), V, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        for (int v = 0; v < V; ++v)
        {
            if (recv_dist[v] != T.dist[v])
            {
                T.dist[v] = recv_dist[v];
                T.affected[v] = true;
            }
            if (recv_parent[v] != T.parent[v] && recv_parent[v] != -1)
                T.parent[v] = recv_parent[v];
        }

        // Check for global convergence
        int local_change_int = local_del_change || local_update_change ? 1 : 0;
        int global_change_int = 0;
        MPI_Allreduce(&local_change_int, &global_change_int, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        global_change = global_change_int != 0;
    }

    // Ensure correctness by iterative convergence (Section 4, Page 4: Avoiding Cycle Formation)
}

// Main function to orchestrate hybrid MPI+OpenMP SSSP computation
int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    string filename = "graph.txt";
    Graph G(0);

    // Load and preprocess graph (Section 5, Page 6: Graph Preprocessing)
    if (!loadGraph(filename, G, rank))
    {
        MPI_Finalize();
        return -1;
    }

    // Compute initial SSSP tree (Section 2, Page 2: Sequential SSSP Initialization)
    SSSPTree T(G.V);
    sequentialSSSP(G, T, 0, rank);

    // Print initial SSSP tree (only rank 0)
    if (rank == 0)
    {
        cout << "\nInitial SSSP Tree (source vertex 1):\n";
        cout << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < G.V; ++i)
        {
            cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                 << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    // Generate edge changes (Section 4, Page 4: Dynamic Graph Changes)
    int num_changes = 5;
    double insert_ratio = 0.0;
    auto changes = generateChanges(G, num_changes, insert_ratio, rank);

    // Print changes (only rank 0)
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

    // Update SSSP tree with dynamic changes (Algorithm 4, Section 5.1, Page 6)
    int async_level = 2; // Section 5.1, Page 6: Asynchronous Updates
    int batch_size = 2;  // Section 5.1, Page 7: Processing Batches of Changes
    parallelSSSPUpdate(G, T, changes, async_level, batch_size, rank, num_processes);

    // Print updated SSSP tree (only rank 0)
    if (rank == 0)
    {
        cout << "\nUpdated SSSP Tree (source vertex 1):\n";
        cout << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < G.V; ++i)
        {
            cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                 << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}