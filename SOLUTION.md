# presentation of the algorithm

The computed algorithm is a greedy nearest neighbors approach, it proceeds as follow:
- check if the graph is connected, if not return -1
- set the current vertex as the vertex of index 0
- while all the edges haven't been visited:
- generate a subgraph of the k nearest neighbours of the current vertex, start with k = 1, if all the edges of the subgraph have been visited, increment k of 1.
- if the subgraph contains unvisited edges:
- find the closest edges to the current vertex
- generate a new path which goes to the closest edge and visits it
- set the current vertex as the one at the end of the path
- actualize the distance, the path and the list of visited edges

