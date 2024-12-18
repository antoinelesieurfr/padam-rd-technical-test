from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import networkx as nx
import numpy as np
import time

# nb will stand for neighbor in the code

class Graph:
    def __init__(self, vertices: list[tuple], edges: list[tuple]):
        """
        Parameters
        ----------
        vertices : list[tuple]
            list of vertices coordinates.
        edges : list[tuple]
            list of edges as tuple (id 1, id 2, weight, coordinates 1, coordinates 2).
        """
        self.vertices = vertices
        self.vertices_df = pd.DataFrame(self.vertices,
                                        columns= ['y','x'])
        self.edges = edges
        self.edges_df = pd.DataFrame(self.edges,
                                     columns = ['id1', 'id2', 'weight', 'coord1', 'coord2'])
        self.G = self.build_Graph()
        self.edges_path = []
        self.visited_edges = []

    def plot(self):
        """
        Plot the graph.
        """
        weights = list(set(edge[2] for edge in self.edges))
        colors = plt.cm.get_cmap("viridis", len(weights))
        _, ax = plt.subplots()
        for i, weight in enumerate(weights):
            lines = [[edge[-2][::-1], edge[-1][::-1]] for edge in self.edges if edge[2] == weight]
            ax.add_collection(LineCollection(lines, colors=colors(i), alpha=0.7, label=f"weight {weight}"))
        ax.plot()
        ax.legend()
        plt.title(f"#E={len(self.edges)}, #V={len(self.vertices)}")
        plt.show()

    def build_Graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        edge_list = self.edges_df[['id1','id2', 'weight']].to_records(index = False).tolist()
        G.add_weighted_edges_from(edge_list)
        return G    
        
    def visit_graph(self) -> float:
        """
        set the list of visited_edges in the order the algorithm visited them,
        the path which visits all the edges and returns the total distance traveled
        """
        if nx.is_connected(self.G) == False:
            print("no pseudo-eulerian path, the graph is not connected")
            return -1
        else:
            #initialization
            edges_path        = []
            visited_edges     = []
            adjacent_vertices = set(self.G.nodes)
            current_vertex    = 0
            traveled_distance = 0
            pruned_graph      = self.G.copy()
           
            while len(visited_edges) < len(self.edges):
                print("%.2f %% done"%(100 * len(visited_edges) / float(len(self.edges))),
                      end ='\r')
                    
                isolated_nodes    = set(nx.isolates(pruned_graph))
                adjacent_vertices = adjacent_vertices - isolated_nodes

                #initiate the loop
                found_new_edge = False
                k              = 0
                while found_new_edge == False:
                    #looks if there is an adjacent vertex at distance k of the current vertex
                    (dijkstra_distances,
                     dijkstra_paths) = nx.single_source_dijkstra(self.G,
                                                                 source = current_vertex,
                                                                 cutoff = k,
                                                                 weight = 'weight')
                    NN_vertices       = set(dijkstra_distances.keys())
                    newfound_vertices = adjacent_vertices & NN_vertices

                    # if there is no adjacent vertex k is incremented
                    if not newfound_vertices:
                        k += 1
                    else:
                        #else we compute the distance to the unvisited vertices of distance k
                        #to the current vertex and choose the first on the list of the closest
                        #vertices
                        found_new_edge = True
                        newfound_dijkstra_distances = {x : dijkstra_distances[x] \
                                                         for x in newfound_vertices}
                        closest_vertex = min(newfound_dijkstra_distances,
                                             key = newfound_dijkstra_distances.get)

                        #we go to the closest adjacent vertex and visit the first
                        #edge on its edges list
                        new_edge = list(pruned_graph.edges(closest_vertex))[0]

                        #actualization
                        next_vertex = list(new_edge)
                        next_vertex.remove(closest_vertex)
                        next_vertex = next_vertex[0]
                        
                        distance = dijkstra_distances[closest_vertex] + \
                            self.G[closest_vertex][next_vertex]['weight']
                        traveled_distance += distance
                        
                        path = dijkstra_paths[closest_vertex] + \
                            [next_vertex]
                        path_edges_list = [(path[x],
                                            path[x + 1]) \
                                           for x in range(len(path) - 1)]
                        edges_path += path_edges_list
                        
                        visited_edges.append(new_edge)
                        #actualize the list of adjacent vertices
                        #(with at least one unvisited edge)
                        pruned_graph.remove_edge(*new_edge)
                        current_vertex = next_vertex
            #save the selected values
            self.edges_path    = edges_path
            self.visited_edges = visited_edges
            return traveled_distance
                      
    def save_path(self, file_name):
        with open(file_name, "w") as output:
            output.write(str(self.edges_path))

        
            

                    
                        
        
    
                


