from input import parse_cmd_line, parse_file
from graph import Graph
import time


def main():
    (in_file,
     plot_graph,
     traverse_graph,
     save)= parse_cmd_line()
    vertices, edges = parse_file(in_file)
    print(f"#E={len(edges)}, #V={len(vertices)}")
    graph = Graph(vertices, edges)
    if plot_graph:
        graph.plot()
    if traverse_graph:
        a= time.time()
        length = graph.visit_graph()
        b = time.time() -a
        print("computed in %.2f seconds"%(b))
        if length == -1:
            print('graph not connected, no pseudo eulerian path')
        else:
            print(f"total length is {length}") 
    graph.save_path(save)

    
    


if __name__ == "__main__":
    main()
