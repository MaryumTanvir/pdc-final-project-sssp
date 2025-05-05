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

// // Algorithm 1: Updates SSSP for a single edge change (Section 2.1, Page 2: Sequential SSSP Update)
// void singleChange(Graph &G, SSSPTree &T, const pair<pair<int, int>, int> &change)
// {
//     int u = change.first.first;
//     int v = change.first.second;
//     int w = change.second;
//     int x, y;

//     // Find the affected vertex (Algorithm 1, lines 3-7)
//     if (T.dist[u] > T.dist[v])
//     {
//         x = v;
//         y = u;
//     }
//     else
//     {
//         x = u;
//         y = v;
//     }

//     // Update graph and SSSP tree (Algorithm 1, lines 10-13)
//     if (w >= 0)
//     {
//         // Edge insertion
//         G.addEdge(u, v, w);
//         int new_dist = safeAdd(T.dist[x], w);
//         if (T.dist[y] > new_dist)
//         {
//             T.dist[y] = new_dist;
//             T.parent[y] = x;
//             T.affected[y] = true;
//         }
//     }
//     else
//     {
//         // Edge deletion
//         G.removeEdge(u, v);
//         if (T.parent[v] == u)
//         {
//             T.dist[v] = INT_MAX;
//             T.parent[v] = -1;
//             T.affected_del[v] = true;
//             T.affected[v] = true;
//         }
//         else if (T.parent[u] == v)
//         {
//             T.dist[u] = INT_MAX;
//             T.parent[u] = -1;
//             T.affected_del[u] = true;
//             T.affected[u] = true;
//         }
//     }

//     // Update affected subgraph (Algorithm 1, lines 15-23)
//     priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
//     if (T.affected[y])
//         pq.push({T.dist[y], y});

//     while (!pq.empty())
//     {
//         int z = pq.top().second;
//         pq.pop();
//         T.affected[z] = false;

//         // Check neighbors for shorter paths
//         int min_dist = T.dist[z];
//         int best_parent = T.parent[z];
//         bool updated = false;

//         for (const auto &edge : G.adj[z])
//         {
//             int n = edge.first;
//             int weight = edge.second;
//             int new_dist = safeAdd(T.dist[n], weight);
//             if (new_dist < min_dist)
//             {
//                 min_dist = new_dist;
//                 best_parent = n;
//                 updated = true;
//             }
//         }

//         if (updated)
//         {
//             T.dist[z] = min_dist;
//             T.parent[z] = best_parent;
//             T.affected[z] = true;
//             for (const auto &edge : G.adj[z])
//             {
//                 if (!T.affected[edge.first])
//                 {
//                     T.affected[edge.first] = true;
//                     pq.push({T.dist[edge.first], edge.first});
//                 }
//             }
//         }
//     }
// }

// Loads graph from file and constructs adjacency list
// (Section 5, Page 6: Graph Preprocessing for Dynamic Updates)
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

    cout << "Loaded graph with " << nvtxs << " vertices and " << edge_set.size() << " edges" << endl;
    return true;
}

// Generates random edge changes for testing dynamic updates
// (Section 4, Page 4: Dynamic Graph Changes)
vector<pair<pair<int, int>, int>> generateChanges(const Graph &G, int num_changes, double insert_ratio)
{
    vector<pair<pair<int, int>, int>> changes;
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
    return changes;
}

// Implements parallel dynamic SSSP update algorithm (Algorithm 4, Section 5.1, Page 6)
// Uses shared-memory parallelism with OpenMP and batch processing
void parallelSSSPUpdate(Graph &G, SSSPTree &T, const vector<pair<pair<int, int>, int>> &changes, int async_level, int batch_size)
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
                        T.affected[u] = true;
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

        // Deletion Phase (Algorithm 4, lines 7-19)
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

                // Disconnect children up to async_level (Section 5.1, Page 6: Asynchronous Updates)
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

                // Mark neighbors as affected (Algorithm 4, lines 38-41)
                for (const auto &edge : G.adj[v])
                {
                    T.affected[edge.first] = true;
                }
            }
        }

        // Update Phase (Algorithm 4, lines 20-end)
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

                    // Check neighbors for shorter paths (Algorithm 4, lines 32-35)
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

                        // Mark neighbors as affected (Algorithm 4, lines 38-41)
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

    // Ensure correctness by iterative convergence (Section 4, Page 4: Avoiding Cycle Formation)
    // The iterative updates ensure distances converge to shortest paths, preventing cycles.
}

// Main function to orchestrate parallel SSSP computation
int main(int argc, char *argv[])
{
    string filename = "graph.txt";
    Graph G(0);

    // Load and preprocess graph (Section 5, Page 6: Graph Preprocessing)
    if (!loadGraph(filename, G))
    {
        return -1;
    }

    // Compute initial SSSP tree (Section 2, Page 2: Sequential SSSP Initialization)
    SSSPTree T(G.V);
    sequentialSSSP(G, T, 0);

    // Print initial SSSP tree
    cout << "\nInitial SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < G.V; ++i)
    {
        cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }

    // Generate edge changes (Section 4, Page 4: Dynamic Graph Changes)
    int num_changes = 5;
    double insert_ratio = 0.0;
    auto changes = generateChanges(G, num_changes, insert_ratio);

    // Print changes
    cout << "\nApplying " << changes.size() << " edge changes (insert_ratio=" << insert_ratio << "):\n";
    for (const auto &change : changes)
    {
        int u = change.first.first + 1;
        int v = change.first.second + 1;
        int w = change.second;
        cout << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
             << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
    }

    // Update SSSP tree with dynamic changes (Algorithm 4, Section 5.1, Page 6)
    int async_level = 2; // Section 5.1, Page 6: Asynchronous Updates
    int batch_size = 2;  // Section 5.1, Page 7: Processing Batches of Changes
    parallelSSSPUpdate(G, T, changes, async_level, batch_size);

    // Print updated SSSP tree
    cout << "\nUpdated SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < G.V; ++i)
    {
        cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i]))
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }

    return 0;
}