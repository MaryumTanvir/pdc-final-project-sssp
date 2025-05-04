#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <limits>

using namespace std;

struct Edge
{
    idx_t from, to, weight;
    char operation; // 'I' for insertion, 'D' for deletion
};

// Dijkstra's algorithm to compute initial SSSP
void dijkstra(idx_t nvtxs, const vector<idx_t> &xadj, const vector<idx_t> &adjncy,
              const vector<idx_t> &adjwgt, idx_t source, vector<idx_t> &dist,
              vector<idx_t> &parent)
{
    vector<bool> visited(nvtxs, false);
    dist.assign(nvtxs, numeric_limits<idx_t>::max());
    parent.assign(nvtxs, -1);
    dist[source] = 0;

    priority_queue<pair<idx_t, idx_t>, vector<pair<idx_t, idx_t>>, greater<>> pq;
    pq.push({0, source});

    while (!pq.empty())
    {
        idx_t d = pq.top().first;
        idx_t u = pq.top().second;
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        for (idx_t i = xadj[u]; i < xadj[u + 1]; ++i)
        {
            idx_t v = adjncy[i];
            idx_t w = adjwgt[i];
            if (!visited[v] && dist[u] + w < dist[v])
            {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}

// Check if edge (u, v) is in the SSSP tree
bool isInSSSPTree(idx_t u, idx_t v, const vector<idx_t> &parent)
{
    return parent[v] == u || parent[u] == v;
}

// Process edge updates (Algorithm 2)
void processUpdates(const vector<Edge> &updates, const vector<idx_t> &dist,
                    vector<idx_t> &parent, vector<bool> &affected,
                    vector<bool> &affectedDel, vector<idx_t> &newDist,
                    vector<idx_t> &newParent, vector<idx_t> &xadj,
                    vector<idx_t> &adjncy, vector<idx_t> &adjwgt)
{
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < updates.size(); ++i)
    {
        idx_t u = updates[i].from - 1; // Convert to 0-based
        idx_t v = updates[i].to - 1;
        idx_t w = updates[i].weight;

        if (updates[i].operation == 'D')
        {
            if (isInSSSPTree(u, v, parent))
            {
                idx_t y = dist[u] > dist[v] ? u : v;
#pragma omp critical
                {
                    newDist[y] = numeric_limits<idx_t>::max();
                    affectedDel[y] = true;
                    affected[y] = true;
                    newParent[y] = -1;
                }
            }
        }
        else if (updates[i].operation == 'I')
        {
            idx_t x = dist[u] < dist[v] ? u : v;
            idx_t y = dist[u] < dist[v] ? v : u;
            if (dist[y] > dist[x] + w)
            {
#pragma omp critical
                {
                    newDist[y] = dist[x] + w;
                    newParent[y] = x;
                    affected[y] = true;
                }
// Update graph (add edge to CSR)
#pragma omp critical
                {
                    adjncy.push_back(v);
                    adjwgt.push_back(w);
                    adjncy.push_back(u);
                    adjwgt.push_back(w);
                }
            }
        }
    }
    // Rebuild xadj after insertions
    xadj.clear();
    xadj.push_back(0);
    for (idx_t i = 1; i <= newDist.size(); ++i)
    {
        xadj.push_back(xadj.back() + count(adjncy.begin(), adjncy.end(), i - 1));
    }
}

// Asynchronous update (Algorithm 4)
void asynchronousUpdate(idx_t nvtxs, const vector<idx_t> &xadj,
                        const vector<idx_t> &adjncy, const vector<idx_t> &adjwgt,
                        vector<idx_t> &dist, vector<idx_t> &parent,
                        vector<bool> &affected, vector<bool> &affectedDel,
                        idx_t asyncLevel)
{
    bool change = true;
    while (change)
    {
        change = false;
// Process deletions
#pragma omp parallel for schedule(dynamic)
        for (idx_t v = 0; v < nvtxs; ++v)
        {
            if (affectedDel[v])
            {
                queue<idx_t> q;
                q.push(v);
                idx_t level = 0;
                while (!q.empty() && level <= asyncLevel)
                {
                    idx_t x = q.front();
                    q.pop();
                    for (idx_t i = xadj[x]; i < xadj[x + 1]; ++i)
                    {
                        idx_t c = adjncy[i];
                        if (parent[c] == x)
                        {
#pragma omp critical
                            {
                                dist[c] = numeric_limits<idx_t>::max();
                                parent[c] = -1;
                                affected[c] = true;
                                affectedDel[c] = true;
                            }
                            q.push(c);
                        }
                    }
                    level++;
                }
                change = true;
            }
        }
        // Process updates
        change = true;
        while (change)
        {
            change = false;
#pragma omp parallel for schedule(dynamic)
            for (idx_t v = 0; v < nvtxs; ++v)
            {
                if (affected[v])
                {
                    affected[v] = false;
                    queue<idx_t> q;
                    q.push(v);
                    idx_t level = 0;
                    while (!q.empty() && level <= asyncLevel)
                    {
                        idx_t x = q.front();
                        q.pop();
                        for (idx_t i = xadj[x]; i < xadj[x + 1]; ++i)
                        {
                            idx_t n = adjncy[i];
                            idx_t w = adjwgt[i];
                            if (dist[x] > dist[n] + w)
                            {
#pragma omp critical
                                {
                                    dist[x] = dist[n] + w;
                                    parent[x] = n;
                                    affected[x] = true;
                                    change = true;
                                }
                                if (level <= asyncLevel)
                                    q.push(x);
                            }
                            if (dist[n] > dist[x] + w)
                            {
#pragma omp critical
                                {
                                    dist[n] = dist[x] + w;
                                    parent[n] = x;
                                    affected[n] = true;
                                    change = true;
                                }
                                if (level <= asyncLevel)
                                    q.push(n);
                            }
                        }
                        level++;
                    }
                }
            }
        }
    }
}

int main()
{
    string datasetFile = "graph.txt";
    string updateFile = "update_new.txt";

    // Read dataset
    ifstream file(datasetFile);
    if (!file.is_open())
    {
        cerr << "Error: Could not open " << datasetFile << endl;
        return -1;
    }

    vector<idx_t> xadj, adjncy, adjwgt;
    idx_t nvtxs = 0, nedges = 0;
    string line;

    while (getline(file, line))
    {
        istringstream iss(line);
        idx_t from, to, weight;
        if (!(iss >> from >> to >> weight))
            break;
        nvtxs = max(nvtxs, max(from, to));
        adjncy.push_back(to - 1);
        adjwgt.push_back(weight);
        adjncy.push_back(from - 1);
        adjwgt.push_back(weight);
        nedges++;
    }
    file.close();

    // Construct xadj
    xadj.push_back(0);
    for (idx_t i = 1; i <= nvtxs; ++i)
    {
        xadj.push_back(xadj.back() + 2 * count(adjncy.begin(), adjncy.end(), i - 1));
    }

    // Compute initial SSSP
    vector<idx_t> dist, parent;
    idx_t source = 0; // Choose source vertex (0-based)
    dijkstra(nvtxs, xadj, adjncy, adjwgt, source, dist, parent);

    // // Print the SSSP Tree
    // cout << "Initial SSSP distances:\n";
    // for (idx_t i = 0; i < nvtxs; ++i)
    // {
    //     cout << "Vertex " << i + 1 << ": Distance = " << dist[i]
    //          << ", Parent = " << (parent[i] == -1 ? -1 : parent[i] + 1) << endl;
    // }
    // cout << "Initial SSSP tree edges:\n";
    // for (idx_t i = 0; i < nvtxs; ++i)
    // {
    //     if (parent[i] != -1)
    //     {
    //         cout << "Edge: " << parent[i] + 1 << " -> " << i + 1 << endl;
    //     }
    // }
    // cout << "Number of vertices: " << nvtxs << ", Number of edges: " << nedges << endl;

    // METIS partitioning

    idx_t ncon = 1, nparts = 2, objval = 0;
    vector<idx_t> part(nvtxs);
    int status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                     NULL, NULL, adjwgt.data(), &nparts,
                                     NULL, NULL, NULL, &objval, part.data());
    if (status == METIS_OK)
    {
        cout << "METIS succeeded. Edge cut: " << objval << endl;
        // for (int i = 0; i < nvtxs; ++i)
        // {
        //     cout << "Vertex " << i + 1 << " in part " << part[i] + 1 << "\n"; // Convert to 1-based index for output
        // }
    }
    else
    {
        cerr << "METIS failed with status: " << status << endl;
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
        idx_t from, to, weight = 0;
        iss >> op >> from >> to;
        if (op == 'I')
            iss >> weight;
        updates.push_back({from, to, weight, op});
    }
    updateIn.close();

    /*
    // Process updates
    vector<bool> affected(nvtxs, false), affectedDel(nvtxs, false);
    vector<idx_t> newDist = dist, newParent = parent;
    processUpdates(updates, dist, parent, affected, affectedDel, newDist, newParent,
                   xadj, adjncy, adjwgt);

    // Asynchronous update
    idx_t asyncLevel = 2; // Adjust based on experiments
    asynchronousUpdate(nvtxs, xadj, adjncy, adjwgt, newDist, newParent,
                       affected, affectedDel, asyncLevel);

    // Output updated SSSP
    cout << "Updated SSSP distances:\n";
    for (idx_t i = 0; i < nvtxs; ++i)
    {
        cout << "Vertex " << i + 1 << ": Distance = " << newDist[i]
             << ", Parent = " << (newParent[i] == -1 ? -1 : newParent[i] + 1) << endl;
    }
    */
    return 0;
}