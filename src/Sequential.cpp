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
    int64_t from, to, weight;
    Edge(int64_t f, int64_t t, int64_t w) : from(f), to(t), weight(w) {}
};

// Graph representation using adjacency list
struct Graph
{

    int64_t V; // Number of vertices
    vector<vector<pair<int64_t, int64_t>>> adj; // Adjacency list: (neighbor, weight)
    Graph(int64_t vertices) : V(vertices), adj(vertices) {}
    void addEdge(int64_t u, int64_t v, int64_t w) {
        if (u >= V || v >= V || u < 0 || v < 0) {
            cerr << "Invalid edge: u=" << u << ", v=" << v << ", V=" << V << endl;
            exit(1);
        }
        adj[u].push_back({v, w});
        adj[v].push_back({u, w}); // Undirected graph
    }


    void removeEdge(int64_t u, int64_t v)
    {
        // Remove (u,v)
        auto it_u = find_if(adj[u].begin(), adj[u].end(), 
                           [v](const pair<int64_t, int64_t>& e) { return e.first == v; });
        if (it_u != adj[u].end()) adj[u].erase(it_u);
        // Remove (v,u)
        auto it_v = find_if(adj[v].begin(), adj[v].end(), 
                           [u](const pair<int64_t, int64_t>& e) { return e.first == u; });
        if (it_v != adj[v].end()) adj[v].erase(it_v);
    }
};

// SSSP tree structure
struct SSSPTree
{
    vector<int64_t> parent;
    vector<int64_t> dist;
    vector<bool> affected;
    vector<bool> affected_del;
    SSSPTree(int64_t V) : parent(V, -1), dist(V, INT64_MAX), affected(V, false), affected_del(V, false) {}
};


// Safe addition to prevent overflow
int64_t safeAdd(int64_t a, int64_t b) {
    if (a == INT64_MAX || b == INT64_MAX) return INT64_MAX;
    if (b > 0 && a > INT64_MAX - b) return INT64_MAX;
    return a + b;
}

// Dijkstra's algorithm for initial SSSP tree
void sequentialSSSP(const Graph& G, SSSPTree& T, int64_t source) {
    T.dist[source] = 0;
    priority_queue<pair<int64_t, int64_t>, vector<pair<int64_t, int64_t>>, greater<>> pq;
    pq.push({0, source});
    
    while (!pq.empty()) {
        int64_t d = pq.top().first;
        int64_t u = pq.top().second;
        pq.pop();
        
        if (d > T.dist[u]) continue;
        
        for (const auto& edge : G.adj[u]) {
            int64_t v = edge.first;
            int64_t w = edge.second;
            int64_t new_dist = safeAdd(T.dist[u], w); // Safe add to prevent overflow
            if (T.dist[v] > new_dist) {
                T.dist[v] = new_dist;
                T.parent[v] = u;
                pq.push({T.dist[v], v});
            }
        }
    }
}

vector<pair<pair<int64_t, int64_t>, int64_t>> readChangesFromFile(const string& updateFilePath) {
    vector<pair<pair<int64_t, int64_t>, int64_t>> changes;
    ifstream file(updateFilePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open update file: " << updateFilePath << endl;
        return changes;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        char type;
        int64_t u, v, w = -1;

        if (!(iss >> type >> u >> v)) continue;
        u--; v--; // Convert to 0-based indexing

        if (type == 'I') {
            iss >> w;
            changes.push_back({{u, v}, w});
        } else if (type == 'D') {
            changes.push_back({{u, v}, -1});
        }
    }

    file.close();
    return changes;
}

// Algorithm 2
// Sequential SSSP update algorithm
void updateSSSP(Graph& G, SSSPTree& T, const vector<pair<pair<int64_t, int64_t>, int64_t>>& changes) {
    int64_t V = G.V;
    
    // Step 1: Process changed edges
    for (const auto& change : changes)
    {
        int64_t u = change.first.first;
        int64_t v = change.first.second;
        int64_t w = change.second;
        
        if (w >= 0) { // Insertion
            G.addEdge(u, v, w); // Update graph
            int64_t x = (T.dist[u] > T.dist[v]) ? v : u;
            int64_t y = (T.dist[u] > T.dist[v]) ? u : v;
            int64_t new_dist = safeAdd(T.dist[x], w);
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
                T.dist[v] = INT64_MAX;
                T.parent[v] = -1;
                T.affected_del[v] = true;
                T.affected[v] = true;
            } else if (T.parent[u] == v) {
                T.dist[u] = INT64_MAX;
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
        for (int64_t v = 0; v < V; ++v) {
            if (T.affected_del[v]) {
                T.affected_del[v] = false;
                for (int64_t c = 0; c < V; ++c) {
                    if (T.parent[c] == v) {
                        T.dist[c] = INT64_MAX;
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
        for (int64_t v = 0; v < V; ++v) {
            if (T.affected[v]) {
                T.affected[v] = false;
                // Try to find a new path to v
                int64_t min_dist = T.dist[v];
                int64_t best_parent = T.parent[v];
                for (const auto& edge : G.adj[v]) {
                    int64_t n = edge.first;
                    int64_t w = edge.second;
                    int64_t new_dist = safeAdd(T.dist[n], w);
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
bool loadGraph(const string& filename, vector<int64_t>& xadj, vector<int64_t>& adjncy, 
               vector<int64_t>& adj_weight, int64_t& num_vertices, int64_t& nedges, Graph& G) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file: " << filename << endl;
        return false;
    }

    vector<Edge> edges;
    map<int64_t, int64_t> vertex_map; // Maps original vertex IDs to 0-based indices
    set<pair<int64_t, int64_t>> edge_set; // To detect duplicates (stores min(u,v), max(u,v))
    int64_t max_vertex = 0;
    string line;
    int64_t line_count = 0;

    // Read edge list
    while (getline(file, line)) {
        line_count++;
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        int64_t from, to;
        int64_t weight = 1; // Default weight for unweighted graphs
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
        int64_t u = vertex_map[e.from];
        int64_t v = vertex_map[e.to];
        // Use canonical edge representation (min(u,v), max(u,v))
        pair<int64_t, int64_t> edge = {min(u, v), max(u, v)};
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
    vector<int64_t> degree(num_vertices, 0);
    for (const auto& e : processed_edges) {
        degree[e.from]++;
        adjncy.push_back(e.to);
        adj_weight.push_back(e.weight);
    }

    // Construct xadj
    for (int64_t i = 0; i < num_vertices; ++i) {
        xadj.push_back(xadj.back() + degree[i]);
    }

    nedges = adjncy.size() / 2; // Each edge appears twice in adjncy

    // Validate CSR
    if (xadj[num_vertices] != static_cast<int64_t>(adjncy.size())) {
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
    vector<int64_t> xadj, adjncy, adj_weight;
    int64_t num_vertices = 0, nedges = 0;
    Graph G(0); // Will be initialized in loadGraph

    // Load and preprocess graph
    if (!loadGraph(filename, xadj, adjncy, adj_weight, num_vertices, nedges, G)) {
        return -1;
    }

    // Compute initial SSSP tree from source vertex 0 (0-based, corresponds to vertex 1 in graph.txt)
    SSSPTree T(num_vertices);
    sequentialSSSP(G, T, 0);

    // Print64_t initial SSSP tree
    cout << "\nInitial SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int64_t i = 0; i < num_vertices; ++i) {
        cout << i + 1 << "\t" << (T.dist[i] == INT64_MAX ? "INF" : to_string(T.dist[i])) 
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }


    //update file 
    string updateFile = "../data/large_update.txt";
    auto changes = readChangesFromFile(updateFile);


    // Print64_t changes
    cout << "\nApplying " << changes.size() << " edge changes from file:\n";

    for (const auto& change : changes) {
        int64_t u = change.first.first + 1; // 1-based for output
        int64_t v = change.first.second + 1;
        int64_t w = change.second;
        cout << (w >= 0 ? "Insert" : "Delete") << " edge: (" << u << ", " << v << ") "
             << (w >= 0 ? "weight=" + to_string(w) : "") << "\n";
    }

    // Update SSSP tree
    updateSSSP(G, T, changes);

    // Print64_t updated SSSP tree
    cout << "\nUpdated SSSP Tree (source vertex 1):\n";
    cout << "Vertex\tDistance\tParent\n";
    for (int64_t i = 0; i < num_vertices; ++i) {
        cout << i + 1 << "\t" << (T.dist[i] == INT64_MAX ? "INF" : to_string(T.dist[i])) 
             << "\t\t" << (T.parent[i] == -1 ? "NONE" : to_string(T.parent[i] + 1)) << "\n";
    }

    return 0;
}