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

using namespace std;

// Structure to represent an edge
struct Edge {
    int from, to, weight;
    Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{

    int V; // Number of vertices
    vector<vector<pair<int, int>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int vertices) : V(vertices), adj(vertices) {}
    void addEdge(int u, int v, int w) {
        if (u >= V || v >= V || u < 0 || v < 0) {
            cerr << "Invalid edge: u=" << u << ", v=" << v << ", V=" << V << endl;
            exit(1);
        }
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
    }


    void removeEdge(int u, int v)
    {
        // Remove (u,v)
        auto it_u = find_if(adj[u].begin(), adj[u].end(), 
                           [v](const pair<int, int>& e) { return e.first == v; });
        if (it_u != adj[u].end()) adj[u].erase(it_u);
        // Remove (v,u)
        auto it_v = find_if(adj[v].begin(), adj[v].end(), 
                           [u](const pair<int, int>& e) { return e.first == u; });
        if (it_v != adj[v].end()) adj[v].erase(it_v);
    }
};

// SSSP tree structure
struct SSSPTree
{
    vector<int> parent;
    vector<int> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int V) : parent(V, -1), dist(V, INT_MAX), affected(V, false), affected_del(V, false) {}
};


// Safe addition to prevent overflow
int safeAdd(int a, int b) {
    if (a == INT_MAX || b == INT_MAX) return INT_MAX;
    if (b > 0 && a > INT_MAX - b) return INT_MAX;
    return a + b;
}

// Dijkstra's algorithm for initial SSSP tree
void sequentialSSSP(const Graph& G, SSSPTree& T, int source) {
    T.dist[source] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int d = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        
        if (d > T.dist[u]) continue;
        
        for (const auto& edge : G.adj[u]) {
            int v = edge.first;
            int w = edge.second;
            int new_dist = safeAdd(T.dist[u], w); // Safe add to prevent overflow
            if (T.dist[v] > new_dist) {
                T.dist[v] = new_dist;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

// Generate random edge changes
vector<pair<pair<int, int>, int>> generateChanges(const Graph& G, int num_changes, double insert_ratio) {
    vector<pair<pair<int, int>, int>> changes;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> vertex_dist(0, G.V - 1);
    uniform_int_distribution<> weight_dist(1, 100);
    bernoulli_distribution insert_dist(insert_ratio);
    
    for (int i = 0; i < num_changes; ++i) {
        int u = vertex_dist(gen);
        int v = vertex_dist(gen);
        while (u == v) v = vertex_dist(gen);
        bool exists = false;
        for (const auto& edge : G.adj[u]) {
            if (edge.first == v) {
                exists = true;
                break;
            }
        }
        int w = insert_dist(gen) ? weight_dist(gen) : (exists ? -1 : weight_dist(gen));
        changes.push_back({{u, v}, w});
    }
    return changes;
}

// Algorithm 2
// Sequential SSSP update algorithm
void updateSSSP(Graph& G, SSSPTree& T, const vector<pair<pair<int, int>, int>>& changes) {
    int V = G.V;
    
    // Step 1: Process changed edges
    for (const auto& change : changes)
    {
        int u = change.first.first;
        int v = change.first.second;
        int w = change.second;
        
        if (w >= 0) { // Insertion
            G.addEdge(u, v, w); // Update graph
            int x = (T.dist[u] > T.dist[v]) ? v : u;
            int y = (T.dist[u] > T.dist[v]) ? u : v;
            int new_dist = safeAdd(T.dist[x], w);
            if (T.dist[y] > new_dist) {
                T.dist[y] = new_dist;
                T.parent[y] = x;
                T.affected[y] = true;
            }
        } 
        
        else { // Deletion
            G.removeEdge(u, v); // Update graph
            // Check if edge is in the current SSSP tree
            if (T.parent[v] == u) {
                T.dist[v] = INT_MAX;
                T.parent[v] = -1;
                T.affected_del[v] = true;
                T.affected[v] = true;
            } else if (T.parent[u] == v) {
                T.dist[u] = INT_MAX;
                T.parent[u] = -1;
                T.affected_del[u] = true;
                T.affected[u] = true;
            }
        }
    }
    
    // Algo 3
    // Step 2: Update affected subgraphs
    bool change = true;
    while (change)
    {
        change = false;
        
        // Process deletion-affected vertices
        for (int v = 0; v < V; ++v) {
            if (T.affected_del[v]) {
                T.affected_del[v] = false;
                for (int c = 0; c < V; ++c) {
                    if (T.parent[c] == v) {
                        T.dist[c] = INT_MAX;
                        T.parent[c] = -1; //
                        T.affected_del[c] = true;
                        T.affected[c] = true;
                        change = true;
                    }
                }
            }
        }
        
        // Update distances for affected vertices
        bool local_change = false;
        for (int v = 0; v < V; ++v) {
            if (T.affected[v]) {
                T.affected[v] = false;
                // Try to find a new path to v
                int min_dist = T.dist[v];
                int best_parent = T.parent[v];
                for (const auto& edge : G.adj[v]) {
                    int n = edge.first;
                    int w = edge.second;
                    int new_dist = safeAdd(T.dist[n], w);
                    if (new_dist < min_dist) {
                        min_dist = new_dist;
                        best_parent = n;
                        local_change = true;
                    }
                }
                if (min_dist != T.dist[v]) {
                    T.dist[v] = min_dist;
                    T.parent[v] = best_parent;
                    T.affected[v] = true;
                    // Mark neighbors as affected
                    for (const auto& edge : G.adj[v]) {
                        T.affected[edge.first] = true;
                    }
                }
            }
        }
        
        change |= local_change;
    }
}

// Function to load and preprocess graph
bool loadGraph(const string& filename, vector<int>& xadj, vector<int>& adjncy, 
               vector<int>& adj_weight, int& num_vertices, int& nedges, Graph& G) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file: " << filename << endl;
        return false;
    }

    vector<Edge> edges;
    map<int, int> vertex_map; // Maps original vertex IDs to 0-based indices
    set<pair<int, int>> edge_set; // To detect duplicates (stores min(u,v), max(u,v))
    int max_vertex = 0;
    string line;
    int line_count = 0;

    // Read edge list
    while (getline(file, line)) {
        line_count++;
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        int from, to;
        int weight = 1; // Default weight for unweighted graphs
        if (!(iss >> from >> to)) {
            cerr << "Error: Invalid format in line " << line_count << ": " << line << endl;
            return false;
        }
        // Read weight if present
        iss >> weight;

        if (from <= 0 || to <= 0) {
            cerr << "Error: Invalid vertex index in line " << line_count 
                 << ": from=" << from << ", to=" << to << endl;
            return false;
        }
        if (from == to) {
            cout << "Warning: Skipping self-loop in line " << line_count 
                 << ": " << from << " -> " << to << endl;
            continue;
        }

        // Update vertex map and max vertex
        vertex_map[from];
        vertex_map[to];
        max_vertex = max(max_vertex, max(from, to));

        // Store edges (1-based for now)
        edges.emplace_back(from, to, weight);
    }
    file.close();

    // Assign contiguous 0-based indices
    num_vertices = 0;
    for (auto& [orig, new_idx] : vertex_map) {
        new_idx = num_vertices++;
    }

    // Initialize Graph
    G = Graph(num_vertices);

    // Process edges: map to 0-based, remove duplicates
    vector<Edge> processed_edges;
    for (const auto& e : edges) {
        int u = vertex_map[e.from];
        int v = vertex_map[e.to];
        // Use canonical edge representation (min(u,v), max(u,v))
        pair<int, int> edge = {min(u, v), max(u, v)};
        if (edge_set.insert(edge).second) {
            processed_edges.emplace_back(u, v, e.weight);
            processed_edges.emplace_back(v, u, e.weight); // Reverse edge for undirected graph
            G.addEdge(u, v, e.weight); // Add to Graph
        } else {
            cout << "Warning: Skipping duplicate edge: " << e.from << " -> " << e.to << endl;
        }
    }

    // Sort edges by 'from' vertex for CSR construction
    sort(processed_edges.begin(), processed_edges.end(), 
         [](const Edge& a, const Edge& b) { return a.from < b.from; });

    // Build CSR format
    xadj.push_back(0);
    vector<int> degree(num_vertices, 0);
    for (const auto& e : processed_edges) {
        degree[e.from]++;
        adjncy.push_back(e.to);
        adj_weight.push_back(e.weight);
    }

    // Construct xadj
    for (int i = 0; i < num_vertices; ++i) {
        xadj.push_back(xadj.back() + degree[i]);
    }

    nedges = adjncy.size() / 2; // Each edge appears twice in adjncy

    // Validate CSR
    if (xadj[num_vertices] != static_cast<int>(adjncy.size())) {
        cerr << "Error: CSR format invalid: xadj[" << num_vertices << "]=" << xadj[num_vertices] 
             << ", adjncy.size()=" << adjncy.size() << endl;
        return false;
    }

    cout << "Loaded graph with " << num_vertices << " vertices and " << nedges << " edges" << endl;
    return true;
}

int main() {
    string filename = "../data/graph.txt";

    // Graph Variables
    vector<int> xadj, adjncy, adj_weight;
    int num_vertices = 0, nedges = 0;
    Graph G(0); // Will be initialized in loadGraph

    // Load and preprocess graph
    if (!loadGraph(filename, xadj, adjncy, adj_weight, num_vertices, nedges, G)) {
        return -1;
    }

    // Compute initial SSSP tree from source vertex 0 (0-based, corresponds to vertex 1 in graph.txt)
    SSSPTree T(num_vertices);
    sequentialSSSP(G, T, 0);

    // Print initial SSSP tree
    cout << "\nInitial SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < num_vertices; ++i) {
        cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i])) 
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }

    // Generate edge changes
    int num_changes = 5; // Small number for testing
    double insert_ratio = 0.1; // 10% insertions, 90% deletions
    auto changes = generateChanges(G, num_changes, insert_ratio);

    // Print changes
    cout << "\nApplying " << num_changes << " edge changes (insert_ratio=" << insert_ratio << "):\n";
    for (const auto& change : changes) {
        int u = change.first.first + 1; // 1-based for output
        int v = change.first.second + 1;
        int w = change.second;
        cout << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
             << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
    }

    // Update SSSP tree
    updateSSSP(G, T, changes);

    // Print updated SSSP tree
    cout << "\nUpdated SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int i = 0; i < num_vertices; ++i) {
        cout << i + 1 << "\t" << (T.dist[i] == INT_MAX ? "INF" : to_string(T.dist[i])) 
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }


    return 0;
}