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

// Global output file stream
ofstream out_file;

// Macro to print to the output file
#define PRINT(msg)                   \
    do                               \
    {                                \
        if (out_file.is_open())      \
        {                            \
            out_file << msg << endl; \
        }                            \
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
    int V;                              // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices) : V(vertices), adj(vertices) {}
    // Add edge to the graph (undirected)
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

// Safe addition to prevent integer overflow in distance calculations (Safe Distance Updates)
int safeAdd(int a, int b)
{
    if (a == INT_MAX || b == INT_MAX)
        return INT_MAX;
    if (b > 0 && a > INT_MAX - b)
        return INT_MAX;
    return a + b;
};

// Implements Dijkstra's algorithm for initial SSSP tree computation (Sequential SSSP Initialization)
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

// Loads graph from file and constructs adjacency list
// (Graph Preprocessing for Dynamic Updates)
bool loadGraph(const string &filename, Graph &G)
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

    // Read edge list from file
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

    // Assign 0-based indices to vertices
    int nvtxs = 0;
    for (auto &[orig, new_idx] : vertex_map)
    {
        new_idx = nvtxs++;
    }

    // Initialize graph with adjacency list
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
        else
        {
            cout << "Warning: Skipping duplicate edge: " << e.from << " -> " << e.to << endl;
        }
    }

    stringstream ss;
    ss << "Loaded graph with " << nvtxs << " vertices and " << edge_set.size() << " edges";
    PRINT(ss.str());
    return true;
}

// Loads edge changes from a file
// Format: I/D <u> <v> [<weight>] (I for insertion, D for deletion, weight for insertions)
vector<pair<pair<int, int>, int>> loadChanges(const string &filename, const Graph &G)
{
    vector<pair<pair<int, int>, int>> changes;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open changes file: " << filename << endl;
        exit(1);
    }

    string line;
    int line_count = 0;
    set<pair<int, int>> change_set; // Track edges to avoid duplicates

    while (getline(file, line))
    {
        line_count++;
        if (line.empty() || line[0] == '#')
            continue;

        istringstream iss(line);
        char op;
        int u, v, weight = -1; // Default -1 for deletions
        if (!(iss >> op >> u >> v))
        {
            cerr << "Error: Invalid format in line " << line_count << ": " << line << endl;
            exit(1);
        }
        if (op == 'I' && !(iss >> weight))
        {
            cerr << "Error: Missing weight for insertion in line " << line_count << ": " << line << endl;
            exit(1);
        }

        // Convert 1-based to 0-based indices
        u--;
        v--;
        if (u < 0 || u >= G.V || v < 0 || v >= G.V || u == v)
        {
            cerr << "Error: Invalid vertex index in line " << line_count
                 << ": u=" << u + 1 << ", v=" << v + 1 << ", V=" << G.V << endl;
            exit(1);
        }

        // Normalize edge to avoid duplicates (u < v)
        int u_norm = min(u, v);
        int v_norm = max(u, v);
        pair<int, int> edge = {u_norm, v_norm};

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
    return changes;
}

// Implements parallel dynamic SSSP update algorithm (Algorithm 4)
// shared-memory parallelism with OpenMP, batch processing, and redundant computations to avoid locks
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes, int async_level, int batch_size)
{
    int V = G.V;
    vector<pair<pair<int, int>, int>> deletions, insertions;

    // Separate deletions and insertions
    for (const auto &change : changes)
    {
        if (change.second < 0)
            deletions.push_back(change);
        else
            insertions.push_back(change);
    }

    // Process changes in batches (Batch Processing of Changes)
    auto processBatch = [&](const vector<pair<pair<int, int>, int>> &batch, bool is_deletion)
    {
        // Step 1: Apply graph updates sequentially to avoid concurrent modifications
        for (const auto &change : batch)
        {
            int u = change.first.first;
            int v = change.first.second;
            int w = change.second;
            if (w >= 0)
            {
                G.addEdge(u, v, w); // Algorithm 2
            }
            else
            {
                G.removeEdge(u, v); // Algorithm 2
            }
        }

        // Step 2: Identify affected vertices in parallel without locks (Algorithm 2)
        // Allow redundant computations
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < batch.size(); ++i)
        {
            int u = batch[i].first.first;
            int v = batch[i].first.second;
            int w = batch[i].second;

            if (w >= 0)
            {
                // Edge insertion
                int x = (T.dist[u] > T.dist[v]) ? v : u;
                int y = (T.dist[u] > T.dist[v]) ? u : v;
                int new_dist = safeAdd(T.dist[x], w);
                if (T.dist[y] > new_dist)
                {
                    // Allow redundant updates; correctness ensured by iterative convergence
                    T.dist[y] = new_dist;
                    T.parent[y] = x;
                    T.affected[y] = true;
                }
            }
            else
            {
                // Edge deletion
                if (T.parent[v] == u)
                {
                    // Allow redundant updates
                    T.dist[v] = INT_MAX;
                    T.parent[v] = -1;
                    T.affected_del[v] = true;
                    T.affected[v] = true;
                }
                else if (T.parent[u] == v)
                {
                    // Allow redundant updates
                    T.dist[u] = INT_MAX;
                    T.parent[u] = -1;
                    T.affected_del[u] = true;
                    T.affected[u] = true;
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

    // Step 3: Update affected subgraphs (Algorithm 4)
    bool global_change = true;
    while (global_change)
    {
        global_change = false;

        // Deletion Phase (Algorithm 4)
        bool local_del_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_del_change)
        for (int v = 0; v < V; ++v)
        {
            if (T.affected_del[v])
            {
                T.affected_del[v] = false;
                queue<int> Q;
                Q.push(v);
                int level = 0;

                // Disconnect children up to async_level (Asynchronous Updates)
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

                // Mark neighbors as affected (Algorithm 4)
                for (const auto &edge : G.adj[v])
                {
                    T.affected[edge.first] = true;
                }
            }
        }

        // Update Phase (Algorithm 4)
        bool local_update_change = false;
#pragma omp parallel for schedule(dynamic) reduction(| : local_update_change)
        for (int v = 0; v < V; ++v)
        {
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

                    // Check neighbors for shorter paths (Algorithm 4)
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
                        // Allow redundant updates; correctness ensured by iterative convergence
                        T.dist[x] = min_dist;
                        T.parent[x] = best_parent;
                        T.affected[x] = true;
                        local_update_change = true;

                        // Mark neighbors as affected (Algorithm 4)
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

        global_change = local_del_change || local_update_change;
    }

    // Ensure correctness by iterative convergence (Avoiding Cycle Formation)
    // The iterative updates ensure distances converge to shortest paths, preventing cycles.
}

// Main
int main(int argc, char *argv[])
{
<<<<<<< HEAD:src/DynamicPaper.cpp
    string filename = "../data/large_input.txt";
=======
    // Open output file
    out_file.open("results.txt");
    if (!out_file.is_open())
    {
        cout << "Error: Could not open output file: results.txt" << endl;
        return -1;
    }

    string graph_filename = "graph.txt";
    string changes_filename = "update_small.txt";
>>>>>>> 8cf873a1f19c37efe500098f2836c80d7872c5df:src/Dynamic_MP.cpp
    Graph G(0);

    // Load and preprocess graph (Graph Preprocessing)
    if (!loadGraph(graph_filename, G))
    {
        if (out_file.is_open())
            out_file.close();
        return -1;
    }

    // Compute initial SSSP tree (Sequential SSSP Initialization)
    SSSPTree T(G.V);
    sequentialSSSP(G, T, 0);

    // Output initial SSSP tree
    // Terminal: First 10 vertices
    stringstream ss_term;
    ss_term << "\nInitial SSSP Tree (source vertex 1, first 10 vertices):\n"
            << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < min(10, G.V); ++i)
    {
        ss_term << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }
    cout << ss_term.str() << endl; // Terminal only

    // File: All vertices
    if (out_file.is_open())
    {
        out_file << "\nInitial SSSP Tree (source vertex 1, all vertices):\n"
                 << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < G.V; ++i)
        {
            out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                     << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    // Load updates from file
    auto changes = loadChanges(changes_filename, G);

    // Print changes
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
    PRINT(ss.str());

    // Update SSSP tree with dynamic changes (Algorithm 4)
    int async_level = 2; // Asynchronous Updates
    int batch_size = 2;  // Processing Batches of Changes
    parallelSSSPUpdate(G, T, changes, async_level, batch_size);

    // Output updated SSSP tree
    // Terminal: First 10 vertices
    stringstream ss_term_updated;
    ss_term_updated << "\nUpdated SSSP Tree (source vertex 1, first 10 vertices):\n"
                    << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < min(10, G.V); ++i)
    {
        ss_term_updated << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                        << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }
    cout << ss_term_updated.str() << endl; // Terminal only

    // File: All vertices
    if (out_file.is_open())
    {
        out_file << "\nUpdated SSSP Tree (source vertex 1, all vertices):\n"
                 << "Vertex\tDistance\tParent\n";
        for (int i = 0; i < G.V; ++i)
        {
            out_file << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
                     << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
        }
    }

    if (out_file.is_open())
        out_file.close();

    return 0;
}