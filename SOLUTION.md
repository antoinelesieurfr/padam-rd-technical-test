# presentation of the algorithm

The computed algorithm is a greedy nearest neighbors approach, it proceeds as follow:
- check if the graph is connected, if not return -1
- start with the edge of index 0
- go to one vertex of the edge
- check the neighboring vertices of the current vertex:
- if vertices have not been visited yet, go to the vertex whose edge has minimum weight
- if all vertices have been visited, expand the neighborhood to the neighbors of the neighbors and so on until you find an unvisited edge, store the path, add the weights and go to the corresponding vertex
- repeat the procedure until we cross all the edges
We used a DataFrame to index the edges, the result is the indices of the edges DataFrame in the order they have been visited, possibly several times.

