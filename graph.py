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


    def get_nbs_edges(self, vertex_id) -> list[tuple]:
        """
        returns a DataFrame listing vertices indexes connected to the vertex of index vertex_id
        by one edge and their weight
        """
        df_index1 = self.edges_df[self.edges_df.id1 == vertex_id]
        df_index2 = self.edges_df[self.edges_df.id2 == vertex_id]
        tuple_list1 = df_index1[['id2','weight']].to_records(index=True).tolist()
        tuple_list2 = df_index2[['id1','weight']].to_records(index=True).tolist()
        return pd.DataFrame(tuple_list1 + tuple_list2,
                            columns = ['index_edge', 'nbs', 'weight'])
    
    def is_connected(self) -> bool:
        G=nx.Graph()
        G.add_nodes_from(range(len(self.vertices)))
        edge_list1 = self.edges_df[['id1','id2']].to_records(index=False).tolist()
        edge_list2 = self.edges_df[['id2','id1']].to_records(index=False).tolist() #graph is undirected
        G.add_edges_from(edge_list1)
        G.add_edges_from(edge_list2)
        return nx.is_connected(G)


    def visit_graph(self) -> float:
        """
        return the list of edges in the order the algorithm visited them 
        and a total distance traveled
        """
        if self.is_connected() == False:
            print("no eulerian path, the graph is not connected")
            return -1
        else:
            edges_path = [0] # we start at edge 0
            self.visited_edges = [0]
            current_vertex = self.edges_df.iloc[0].id2 # the first vertex is connected to the first edge
            traveled_distance = self.edges_df.iloc[0].weight
            while len(self.visited_edges) < len(self.edges):
                nbs_df = self.get_nbs_edges(current_vertex) 
                unvisited_nbs_edges_df = nbs_df[~nbs_df.index_edge.isin(self.visited_edges)]
                is_empty = False 
                if unvisited_nbs_edges_df.empty:
                    is_empty = True
                    # If all the edges in the nbs of the vertex have been visited,
                    # we expand the nbhood to the nbs of the nbs and so on
                    # until we find an unvisited edge
                    #create a column to store the intermediate path
                    nbs_df['inter_edges'] = np.empty((len(nbs_df), 0)).tolist()
                    while unvisited_nbs_edges_df.empty:
                        new_nbs_df = nbs_df.copy()
                        for i in range(len(nbs_df)): #TODO parallelize
                            # add the nbs of the i-th nb to the nbhood 
                            extended_nbs = self.get_nbs_edges(nbs_df.iloc[i].nbs)
                            #update the weights
                            extended_nbs['weight'] = extended_nbs['weight'] + nbs_df.iloc[i].weight
                            # store the intermediate edge
                            extended_nbs['inter_edges'] = pd.Series([nbs_df.iloc[i].inter_edges] * len(extended_nbs))
                            extended_nbs['inter_edges'] = extended_nbs['inter_edges'].apply(lambda x : x + [nbs_df.iloc[i].index_edge]) #TODO find a more efficient way to expand the list in every cell
                            #update the new neighborhood dataframe
                            new_nbs_df = pd.concat([new_nbs_df, extended_nbs]) 
                        nbs_df = new_nbs_df
                        #remove duplicates
                        nbs_df = nbs_df.sort_values('weight', ascending=False).drop_duplicates('nbs') #TODO check if it really picks the smallest value
                        #print(nbs_df, end = '\r')
                        unvisited_nbs_edges_df = nbs_df[~nbs_df.index_edge.isin(self.visited_edges)]
                        #print(unvisited_nbs_edges_df, end='\r')

                #find the nearset_nbs
                nearest_nbs_edges_df = unvisited_nbs_edges_df[unvisited_nbs_edges_df.weight == unvisited_nbs_edges_df.weight.min()]
                #select the first of the nearest nb in the list
                selected_nb = nearest_nbs_edges_df.iloc[0]
                self.visited_edges = self.visited_edges + [selected_nb.index_edge]
                print(len(self.visited_edges)/float(len(self.edges)), end='\r')
                if is_empty:
                    edges_path = edges_path + selected_nb.inter_edges + [selected_nb.index_edge]
                else:
                    edges_path = edges_path + [selected_nb.index_edge]
                current_vertex = selected_nb.nbs
                traveled_distance = traveled_distance + selected_nb.weight
            self.edges_path = edges_path
            print("traveled distance is %s"%traveled_distance)
            return traveled_distance

    def save_path(self):
        with open("result.txt", "w") as output:
            output.write(str(self.edges_path))

        
            

                    
                        
        
    
                


