#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <queue>
#include <set>
#include <limits>
#include <algorithm>

using namespace std;

struct Edge
{
    int64_t from, to, weight;
    char operation; // 'I' for insertion, 'D' for deletion
};

// Graph representation: adjacency list for full graph and SSSP tree
using AdjList = vector<vector<pair<int64_t, int64_t>>>; // {vertex, weight}

// Dijkstra's algorithm to compute initial SSSP
void dijkstra(int64_t nvtxs, const AdjList &graph, int64_t source,
              vector<int64_t> &dist, vector<int64_t> &parent, AdjList &ssspTree)
{
    vector<bool> visited(nvtxs, false);
    dist.assign(nvtxs, numeric_limits<int64_t>::max());
    parent.assign(nvtxs, -1);
    dist[source] = 0;
    ssspTree.assign(nvtxs, {});

    priority_queue<pair<int64_t, int64_t>, vector<pair<int64_t, int64_t>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        int64_t d = pq.top().first;
        int64_t u = pq.top().second;
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        for (const auto &[v, w] : graph[u])
        {
            if (!visited[v] && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }

    // Build SSSP tree adjacency list
    for (int64_t v = 0; v < nvtxs; ++v)
    {
        if (parent[v] != -1)
        {
            int64_t u = parent[v];
            // Find weight of edge (u, v) in graph
            for (const auto &[neighbor, w] : graph[u])
            {
                if (neighbor == v)
                {
                    ssspTree[u].push_back({v, w});
                    ssspTree[v].push_back({u, w}); // Undirected tree edge
                    break;
                }
            }
        }
    }
}

// Check if edge (u, v) is in the SSSP tree
bool isInSSSPTree(int64_t u, int64_t v, const vector<int64_t> &parent)
{
    return parent[v] == u || parent[u] == v;
}

// Algorithm 2: Identify affected vertices
void processUpdates(const vector<Edge> &updates, const vector<int64_t> &dist,
                    const vector<int64_t> &parent, const AdjList &graph,
                    vector<bool> &affected, vector<bool> &affectedDel,
                    vector<int64_t> &newDist, vector<int64_t> &newParent,
                    AdjList &newGraph)
{
    newDist = dist;
    newParent = parent;
    affected.assign(dist.size(), false);
    affectedDel.assign(dist.size(), false);
    newGraph = graph; // Copy graph for updates
    set<pair<int64_t, int64_t>> deletedEdges;

    for (const auto &update : updates)
    {
        int64_t u = update.from - 1; // 0-based
        int64_t v = update.to - 1;
        int64_t w = update.weight;

        if (update.operation == 'D')
        {
            // Remove edge from newGraph
            newGraph[u].erase(
                remove_if(newGraph[u].begin(), newGraph[u].end(),
                          [v](const auto &e)
                          { return e.first == v; }),
                newGraph[u].end());
            newGraph[v].erase(
                remove_if(newGraph[v].begin(), newGraph[v].end(),
                          [u](const auto &e)
                          { return e.first == u; }),
                newGraph[v].end());
            deletedEdges.insert({min(u, v), max(u, v)});

            if (isInSSSPTree(u, v, parent))
            {
                int64_t y = dist[u] > dist[v] ? u : v; // Child in SSSP tree
                newDist[y] = numeric_limits<int64_t>::max();
                newParent[y] = -1;
                affectedDel[y] = true;
                affected[y] = true;
            }
        }
        else if (update.operation == 'I')
        {
            // Add or update edge in newGraph
            bool updated = false;
            for (auto &e : newGraph[u])
            {
                if (e.first == v)
                {
                    e.second = min(e.second, w); // Update to minimum weight
                    updated = true;
                    break;
                }
            }
            if (!updated)
            {
                newGraph[u].push_back({v, w});
            }
            updated = false;
            for (auto &e : newGraph[v])
            {
                if (e.first == u)
                {
                    e.second = min(e.second, w);
                    updated = true;
                    break;
                }
            }
            if (!updated)
            {
                newGraph[v].push_back({u, w});
            }

            // Check if insertion affects distances
            int64_t x = dist[u] < dist[v] ? u : v;
            int64_t y = dist[u] < dist[v] ? v : u;
            if (dist[y] > dist[x] + w)
            {
                newDist[y] = dist[x] + w;
                newParent[y] = x;
                affected[y] = true;
            }
        }
    }

    // Print affected vertices and updated graph
    cout << "After processing updates:\n";
    cout << "Affected vertices: ";
    for (size_t i = 0; i < affected.size(); ++i)
    {
        if (affected[i])
            cout << i + 1 << " ";
    }
    cout << "\nAffected by deletions: ";
    for (size_t i = 0; i < affectedDel.size(); ++i)
    {
        if (affectedDel[i])
            cout << i + 1 << " ";
    }
    cout << "\nNew graph adjacency list:\n";
    for (size_t u = 0; u < newGraph.size(); ++u)
    {
        if (!newGraph[u].empty())
        {
            cout << "Vertex " << u + 1 << ": ";
            for (const auto &[v, w] : newGraph[u])
            {
                cout << "(" << v + 1 << ", " << w << ") ";
            }
            cout << "\n";
        }
    }
}

// Algorithm 3: Update SSSP tree
void updateSSSP(int64_t nvtxs, const AdjList &graph, vector<int64_t> &dist,
                vector<int64_t> &parent, vector<bool> &affected,
                vector<bool> &affectedDel, AdjList &ssspTree)
{
    bool change = true;
    while (change)
    {
        change = false;

        // Step 1: Process deletions (disconnect affected vertices)
        for (int64_t v = 0; v < nvtxs; ++v)
        {
            if (affectedDel[v])
            {
                queue<int64_t> q;
                q.push(v);
                while (!q.empty())
                {
                    int64_t x = q.front();
                    q.pop();
                    for (const auto &[c, _] : ssspTree[x])
                    {
                        if (parent[c] == x)
                        {
                            dist[c] = numeric_limits<int64_t>::max();
                            parent[c] = -1;
                            affected[c] = true;
                            affectedDel[c] = true;
                            q.push(c);
                        }
                    }
                }
                change = true;
            }
        }

        // Clear SSSP tree for affected vertices
        for (int64_t v = 0; v < nvtxs; ++v)
        {
            if (affectedDel[v])
            {
                ssspTree[v].clear();
            }
        }

        // Step 2: Process updates (relax edges)
        change = true;
        while (change)
        {
            change = false;
            for (int64_t v = 0; v < nvtxs; ++v)
            {
                if (affected[v])
                {
                    affected[v] = false;
                    queue<int64_t> q;
                    q.push(v);
                    while (!q.empty())
                    {
                        int64_t x = q.front();
                        q.pop();
                        for (const auto &[n, w] : graph[x])
                        {
                            if (dist[x] > dist[n] + w)
                            {
                                dist[x] = dist[n] + w;
                                parent[x] = n;
                                affected[x] = true;
                                change = true;
                                q.push(x);
                            }
                            if (dist[n] > dist[x] + w)
                            {
                                dist[n] = dist[x] + w;
                                parent[n] = x;
                                affected[n] = true;
                                change = true;
                                q.push(n);
                            }
                        }
                    }
                }
            }
        }
    }

    // Rebuild SSSP tree
    ssspTree.assign(nvtxs, {});
    for (int64_t v = 0; v < nvtxs; ++v)
    {
        if (parent[v] != -1)
        {
            int64_t u = parent[v];
            for (const auto &[neighbor, w] : graph[u])
            {
                if (neighbor == v)
                {
                    ssspTree[u].push_back({v, w});
                    ssspTree[v].push_back({u, w});
                    break;
                }
            }
        }
    }
}

int main()
{
    string datasetFile = "graph.txt";
    string updateFile = "update_small.txt";

    // Read dataset
    ifstream file(datasetFile);
    if (!file.is_open())
    {
        cerr << "Error: Could not open " << datasetFile << endl;
        return -1;
    }

    int64_t nvtxs = 0, nedges = 0;
    string line;
    set<pair<int64_t, int64_t>> edgeSet;
    AdjList graph;

    while (getline(file, line))
    {
        istringstream iss(line);
        int64_t from, to, weight;
        if (!(iss >> from >> to >> weight))
            break;
        nvtxs = max(nvtxs, max(from, to));
        edgeSet.insert({min(from - 1, to - 1), max(from - 1, to - 1)});
        nedges++;
    }
    file.close();

    graph.resize(nvtxs);
    file.open(datasetFile);
    while (getline(file, line))
    {
        istringstream iss(line);
        int64_t from, to, weight;
        if (!(iss >> from >> to >> weight))
            break;
        from--;
        to--; // 0-based
        graph[from].push_back({to, weight});
        graph[to].push_back({from, weight}); // Undirected
    }
    file.close();

    cout << "Number of vertices: " << nvtxs << "\nNumber of edges: " << nedges << "\n";
    cout << "Initial graph adjacency list:\n";
    for (int64_t u = 0; u < nvtxs; ++u)
    {
        if (!graph[u].empty())
        {
            cout << "Vertex " << u + 1 << ": ";
            for (const auto &[v, w] : graph[u])
            {
                cout << "(" << v + 1 << ", " << w << ") ";
            }
            cout << "\n";
        }
    }

    // Compute initial SSSP
    vector<int64_t> dist, parent;
    AdjList ssspTree;
    int64_t source = 0; // Vertex 1 (0-based)
    dijkstra(nvtxs, graph, source, dist, parent, ssspTree);

    cout << "Initial SSSP distances:\n";
    for (int64_t i = 0; i < nvtxs; ++i)
    {
        cout << "Vertex " << i + 1 << ": Distance = " << (dist[i] == numeric_limits<int64_t>::max() ? "INF" : to_string(dist[i]))
             << ", Parent = " << (parent[i] == -1 ? -1 : parent[i] + 1) << "\n";
    }
    cout << "Initial SSSP tree adjacency list:\n";
    for (int64_t u = 0; u < nvtxs; ++u)
    {
        if (!ssspTree[u].empty())
        {
            cout << "Vertex " << u + 1 << ": ";
            for (const auto &[v, w] : ssspTree[u])
            {
                cout << "(" << v + 1 << ", " << w << ") ";
            }
            cout << "\n";
        }
    }

    // Read updates
    vector<Edge> updates;
    ifstream updateIn(updateFile);
    if (!updateIn.is_open())
    {
        cerr << "Error: Could not open " << updateFile << endl;
        return -1;
    }
    while (getline(updateIn, line))
    {
        istringstream iss(line);
        char op;
        int64_t from, to, weight = 0;
        iss >> op >> from >> to;
        if (op == 'I')
            iss >> weight;
        updates.push_back({from, to, weight, op});
    }
    updateIn.close();

    cout << "Number of updates: " << updates.size() << "\nUpdates:\n";
    for (const auto &update : updates)
    {
        cout << "Operation: " << update.operation << ", From: " << update.from
             << ", To: " << update.to;
        if (update.operation == 'I')
            cout << ", Weight: " << update.weight;
        cout << "\n";
    }

    // Process updates (Algorithm 2)
    vector<bool> affected, affectedDel;
    vector<int64_t> newDist, newParent;
    AdjList newGraph;
    processUpdates(updates, dist, parent, graph, affected, affectedDel, newDist, newParent, newGraph);

    // Update SSSP (Algorithm 3)
    updateSSSP(nvtxs, newGraph, newDist, newParent, affected, affectedDel, ssspTree);

    cout << "After update:\nUpdated SSSP distances:\n";
    for (int64_t i = 0; i < nvtxs; ++i)
    {
        cout << "Vertex " << i + 1 << ": Distance = " << (newDist[i] == numeric_limits<int64_t>::max() ? "INF" : to_string(newDist[i]))
             << ", Parent = " << (newParent[i] == -1 ? -1 : newParent[i] + 1) << "\n";
    }
    cout << "Updated SSSP tree adjacency list:\n";
    for (int64_t u = 0; u < nvtxs; ++u)
    {
        if (!ssspTree[u].empty())
        {
            cout << "Vertex " << u + 1 << ": ";
            for (const auto &[v, w] : ssspTree[u])
            {
                cout << "(" << v + 1 << ", " << w << ") ";
            }
            cout << "\n";
        }
    }

    return 0;
}