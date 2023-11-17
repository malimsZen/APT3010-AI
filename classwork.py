
from Graph import Graph, Vertex
from Queue import Queue


def breadth_first_search(graph, start_vertex, distances = {}):
    discovered_set = set()
    frontier_queue = Queue()
    visited_list = []
    
    distances[start_vertex] = 0 # start vertex has a distance of 0 from itself
    
    frontier_queue.enqueue(start_vertex)
    discovered_set.add(start_vertex)
    
    while (frontier_queue.list.head != None):
        current_vertex = frontier_queue.dequeue()
        visited_list.append(current_vertex)
        for adjacent_vertex in graph.adjancency_list[current_vertex]:
            if adjacent_vertex not in discovered_set:
                frontier_queue.enqueue(adjacent_vertex)
                discovered_set.add(adjacent_vertex)
                
                distances[adjacent_vertex] = distances[current_vertex] + 1
    return visited_list


g = Graph()
vertex_a  =  Vertex('Joe')
vertex_b  =  Vertex('Eva')
vertex_c  =  Vertex('Taj')
vertex_d  =  Vertex('Chen')
vertex_e  =  Vertex('Lily')
vertex_f  =  Vertex('Jun')
vertex_g  =  Vertex('Ken')

vertices = [vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f, vertex_g]

for vertex in vertices:
    g.add_vertex(vertex)
    g.add_undirected_edge(vertex_a, vertex_c)
    g.add_undirected_edge(vertex_b, vertex_e)
    g.add_undirected_edge(vertex_c, vertex_d)
    g.add_undirected_edge(vertex_c, vertex_e)
    g.add_undirected_edge(vertex_d, vertex_f)
    g.add_undirected_edge(vertex_e, vertex_f)
    g.add_undirected_edge(vertex_f, vertex_g)


start_name = input("Enter starting person's name")
print()


star_vertex = None

for vertex in vertices:
    if vertex.label == start_name:
        start_vertex = vertex
        
if start_vertex is None:
    print(f"Start vertex not found {start_name}")
else:
    vertex_distances = {}
    visited_list = breadth_first_search(g, start_vertex, vertex_distances)
    
    print("Breadth-first search transerval")
    print(f"start vertex {start_vertex.label}")
    for vertex in visited_list:
        print(f"{vertex.label} : {vertex_distances[vertex]}")

g = Graph()
vertex_a  =  Vertex('A')
vertex_b  =  Vertex('B')
vertex_c  =  Vertex('C')
vertex_d  =  Vertex('D')
vertex_e  =  Vertex('E')
vertex_f  =  Vertex('F')
vertex_g  =  Vertex('G')
vertex_h  =  Vertex('H')
vertex_i  =  Vertex('I')
vertex_j  =  Vertex('J')


vertices = [vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f, vertex_g, vertex_h, vertex_i, vertex_j]

for vertex in vertices:
    g.add_vertex(vertex)
    g.add_undirected_edge(vertex_a, vertex_b)
    g.add_undirected_edge(vertex_b, vertex_c)
    g.add_undirected_edge(vertex_b, vertex_f)
    g.add_undirected_edge(vertex_c, vertex_d)
    g.add_undirected_edge(vertex_c, vertex_g)
    g.add_undirected_edge(vertex_d, vertex_g)
    g.add_undirected_edge(vertex_d, vertex_h)
    g.add_undirected_edge(vertex_e, vertex_b)
    g.add_undirected_edge(vertex_e, vertex_f)
    g.add_undirected_edge(vertex_e, vertex_i)
    g.add_undirected_edge(vertex_f, vertex_c)
    g.add_undirected_edge(vertex_f, vertex_i)
    g.add_undirected_edge(vertex_g, vertex_h)
    g.add_undirected_edge(vertex_g, vertex_j)



start_name = input("Enter server's name")
print()


star_vertex = None

for vertex in vertices:
    if vertex.label == start_name:
        start_vertex = vertex
        
if start_vertex is None:
    print(f"Start vertex not found {start_name}")
else:
    vertex_distances = {}
    visited_list = breadth_first_search(g, start_vertex, vertex_distances)
    
    print("Breadth-first search transerval")
    print(f"start vertex {start_vertex.label}")
    for vertex in visited_list:
        print(f"{vertex.label} : {vertex_distances[vertex]}")


from Graph import Vertex, Graph


def depth_first_search(graph, start_vertex, visited_func):
    vertex_stack = [start_vertex]
    visited_set = set()
    
    while len(vertex_stack) > 0:
        current_vertex = vertex_stack.pop()
        if current_vertex not in visited_set:
            visited_func(current_vertex)
            visited_set.add(current_vertex)
            for adjacent_vertex in graph.adjancency_list[current_vertex]:
                vertex_stack.append(adjacent_vertex)

vertex_names = ['A', 'B', 'C', 'D', 'E', 'F']

graph1 = Graph()
graph2 = Graph()
graph3 = Graph()
graphs = [graph1, graph2, graph3]

for vertex_name in vertex_names:
    for graph in graphs:
        graph.add_vertex(Vertex(vertex_name))
        

# graph1's edges
graph1.add_undirected_edge(graph1.get_vertex("A"), graph1.get_vertex("B"))
graph1.add_undirected_edge(graph1.get_vertex("A"), graph1.get_vertex("D"))
graph1.add_undirected_edge(graph1.get_vertex("B"), graph1.get_vertex("E"))
graph1.add_undirected_edge(graph1.get_vertex("B"), graph1.get_vertex("F"))
graph1.add_undirected_edge(graph1.get_vertex("C"), graph1.get_vertex("F"))
graph1.add_undirected_edge(graph1.get_vertex("E"), graph1.get_vertex("F"))

# graph2's edges
graph2.add_undirected_edge(graph2.get_vertex("A"), graph2.get_vertex("B"))
graph2.add_undirected_edge(graph2.get_vertex("B"), graph2.get_vertex("C"))
graph2.add_undirected_edge(graph2.get_vertex("C"), graph2.get_vertex("F"))
graph2.add_undirected_edge(graph2.get_vertex("D"), graph2.get_vertex("E"))
graph2.add_undirected_edge(graph2.get_vertex("E"), graph2.get_vertex("F"))

# graph3's edges
graph3.add_undirected_edge(graph3.get_vertex("A"), graph3.get_vertex("B"))
graph3.add_undirected_edge(graph3.get_vertex("A"), graph3.get_vertex("E"))
graph3.add_undirected_edge(graph3.get_vertex("B"), graph3.get_vertex("C"))
graph3.add_undirected_edge(graph3.get_vertex("B"), graph3.get_vertex("E"))
graph3.add_undirected_edge(graph3.get_vertex("C"), graph3.get_vertex("E"))
graph3.add_undirected_edge(graph3.get_vertex("D"), graph3.get_vertex("E"))
graph3.add_undirected_edge(graph3.get_vertex("E"), graph3.get_vertex("F"))

visitor = lambda x: print(x.label, end = ' ')

start_vertex_label = "A"

for i in range(0, len(graphs)):
    print(f"Graph {i+1} : ", end="")
    depth_first_search(graphs[i], graphs[i].get_vertex(start_vertex_label), visitor)
    print("\n")





