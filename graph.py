from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import networkx as nx
import numpy as np

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
        G=nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        edge_list = self.edges_df[['id1','id2', 'weight']].to_records(index=False).tolist()
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
            edges_path = []
            visited_edges = []
            current_vertex = 0
            traveled_distance = 0
            while len(visited_edges) < len(self.edges):
                print("%.2f %% done"%(100 * len(visited_edges) / float(len(self.edges))), end ='\r')
                found_new_edge = False
                k = 1
                while found_new_edge == False:
                    #generate the graph of the k nearest neighbours of the current vertex
                    sub_graph = nx.ego_graph(self.G, current_vertex, radius = k)
                    sub_graph_copy = sub_graph.copy()
                    sub_graph_copy.remove_edges_from(visited_edges)
                    if nx.is_empty(sub_graph_copy):
                        #if the subgraph contains only visited edges we look for the k+1
                        #nearest neighbors
                        k += 1
                    else:
                        found_new_edge = True
                        new_edges = np.array(sub_graph_copy.edges)
                        distance = np.inf
                        #for every unvisited edge, we look for the closest path from the current
                        #vertex to the edge by the to adjacent vertices 
                        for i in range(len(new_edges)):
                            distance_inter_right = nx.shortest_path_length(sub_graph,
                                                                           source = current_vertex,
                                                                           weight = 'weight',
                                                                           target = new_edges[i][0])
                            distance_inter_left = nx.shortest_path_length(sub_graph,
                                                                          source = current_vertex,
                                                                          weight = 'weight',
                                                                          target = new_edges[i][1])
                            if distance_inter_left > distance_inter_right:
                                closest_vertex = new_edges[i][0]
                                next_vertex = new_edges[i][1]
                                distance_inter = distance_inter_right
                            else:
                                closest_vertex = new_edges[i][1]
                                next_vertex = new_edges[i][0]
                                distance_inter = distance_inter_left                            
                            #when the distance to the edge if the lowest of the for loop
                            #we select the closest path that visit the edge, store the path
                            #and the distance and set the next current vertex
                            if distance_inter < distance:
                                next_visited_edge = list(new_edges[i])
                                distance = distance_inter
                                path_inter = nx.shortest_path(sub_graph,
                                                              source = current_vertex,
                                                              weight = 'weight',
                                                              target = closest_vertex)
                                path_inter = path_inter + [next_vertex]
                                final_distance = distance_inter + \
                                    self.G[closest_vertex][next_vertex]['weight']
                                next_chosen_vertex = next_vertex

                        path_inter_list = [(path_inter[x],\
                                            path_inter[x + 1]) \
                                           for x in range(len(path_inter) - 1)]
                        #actualization of the edges path, the distance and the current vertex
                        edges_path += path_inter_list
                        visited_edges.append(next_visited_edge)
                        current_vertex = next_chosen_vertex
                        traveled_distance += final_distance
            self.edges_path = edges_path
            self.visited_edges = visited_edges
            return traveled_distance
                        
    def save_path(self, file_name):
        with open(file_name, "w") as output:
            output.write(str(self.edges_path))

        
            

                    
                        
        
    
                


