import numpy as np
import networkx as nx
from A_star import A_star
import matplotlib.pyplot as plt

def main():
    a_star = A_star()

    # Define 30 commonly known locations in Manipal with coordinates
    manipal_locations = {
        "MIT": (15, 55), "Marena": (15, 70), "End Point": (40, 90), "Tiger Circle": (40, 50),
        "Manipal Lake": (60, 80), "Kamath Circle": (25, 60), "KMC": (50, 50), "SOC": (35, 40),
        "MSAP": (20, 30), "DOC": (45, 25), "Deetee": (55, 30), "Hangyo": (70, 60),
        "Dollops": (60, 65), "China Valley": (65, 50), "TC Canteen": (42, 40), "Health Center": (48, 72),
        "Library": (38, 60), "Manipal Greens": (28, 85), "Valley View": (70, 40),
        "16th Block": (10, 50), "8th Block": (20, 45), "AB5": (30, 35), "NLH": (25, 25),
        "FC2": (40, 20), "Pharmacy Block": (55, 15), "Swimming Pool": (60, 35), 
        "Innovation Center": (75, 25), "Student Plaza": (80, 50), "MIT Cafeteria": (85, 70)
    }

    # Create graph
    G = nx.Graph()
    positions = {i: manipal_locations[name] for i, name in enumerate(manipal_locations.keys())}
    node_names = list(manipal_locations.keys())

    # Add nodes
    for i, name in enumerate(node_names):
        G.add_node(i, name=name, pos=positions[i])

    # Define edges
    edges = [
        ("MIT", "16th Block"), ("MIT", "Marena"), ("MIT", "Kamath Circle"), ("16th Block", "8th Block"), 
        ("8th Block", "AB5"), ("AB5", "NLH"), ("NLH", "FC2"), ("FC2", "Pharmacy Block"),
        ("Marena", "Manipal Greens"), ("Manipal Greens", "End Point"), ("Kamath Circle", "Tiger Circle"),
        ("Tiger Circle", "KMC"), ("Tiger Circle", "DOC"), ("KMC", "Health Center"), ("KMC", "Dollops"), 
        ("Dollops", "China Valley"), ("China Valley", "Valley View"), ("Valley View", "Hangyo"), 
        ("Hangyo", "Student Plaza"), ("Student Plaza", "MIT Cafeteria"), ("MIT Cafeteria", "Manipal Lake"), 
        ("Manipal Lake", "End Point"), ("Health Center", "Library"), ("Library", "TC Canteen"), 
        ("TC Canteen", "DOC"), ("DOC", "Deetee"), ("Deetee", "Swimming Pool"), 
        ("Swimming Pool", "Innovation Center"), ("Innovation Center", "Pharmacy Block"), ("Pharmacy Block", "FC2"),
        ("End Point", "Manipal Greens"),("End Point", "Manipal Lake"), ("MSAP", "8th Block")
    ]

    #Makes a graph more suitable for us combining edges and manipal_locations

    #graph_neighbor_coordinates = {"Place": [["Neighbors"],["Neighbor coordinates"]]}  
    graph_neighbor_coordinates = {}
    #Hard code this that is, get the output and store it raw
    for place in edges:
        array_neighbors = []
        array_neighbors_coordinates = []
        for x in edges:
            if(str(x[0]) == str(place[0])): 
                array_neighbors.append(x[1])
                array_neighbors_coordinates.append(manipal_locations[x[1]])
        graph_neighbor_coordinates[place[0]] = [ array_neighbors, array_neighbors_coordinates]


    graph_neighbor_coordinates_keys_to_index = dict()
    for key, i in zip(graph_neighbor_coordinates, range(len(graph_neighbor_coordinates))):
        graph_neighbor_coordinates_keys_to_index[key] = i

    ch = 'y'
    while ch == 'y':
        # Get user input
        print("\nAvailable Locations:\n")
        print(", ".join(node_names))
        start_name = input("\nEnter start location: ").strip()
        end_name = input("Enter end location: ").strip()
        TIME = int(input("Enter time (in minutes from midnight): ").strip())

        # Validate input
        if start_name not in node_names or end_name not in node_names:
            print("Invalid locations! Please check the list and try again.")
            exit()

        start = node_names.index(start_name)
        end = node_names.index(end_name)

        # Convert location names to node indices and add edges with weights
        for loc1, loc2 in edges:
            node1 = node_names.index(loc1)
            node2 = node_names.index(loc2)
            G.add_edge(node1, node2, weight=a_star.cost_at_time(node1, node2, TIME))

        # Define heuristic values (replace with actual heuristics)
        heuristic_values = {i: a_star.cost_at_time(i, end, TIME) for i in G.nodes}

        ######################################################################################
        # Compute shortest path
        #path = nx.shortest_path(G, source=start, target=end, weight='weight')

        # Create dictionary graph for a_star
        graph_dict = {}
        for node in G.nodes:
            name = node_names[node]
            neighbors = {}
            for neighbor in G[node]:
                neighbor_name = node_names[neighbor]
                weight = G[node][neighbor]['weight']
                neighbors[neighbor_name] = weight
            graph_dict[name] = neighbors

        # Run custom A* algorithm
        path_names = a_star.a_star(graph_dict, start_name, end_name, graph_neighbor_coordinates_keys_to_index, TIME % 1440)
        path = [node_names.index(name) for name in path_names]

        # Check if path is found
        if isinstance(path_names, str):
            print(path_names)
            exit()

        # Convert name-based path to index-based path for plotting
        path = [node_names.index(name) for name in path_names]


        ########################################################################
        #path2 = a_Star.a_star(G, origin=start, destination=end)
        #path = [node_names.index(name) for name in path_names]
        ########################################################################

        ########################################################################
        path_cost = round(sum(G[u][v]['weight'] for u, v in zip(path, path[1:])), 2)
        # Create figure
        _, ax = plt.subplots(figsize=(10, 8))

        # Define color map for nodes
        node_colors = []
        for node in G.nodes:
            if node == start:
                node_colors.append('lime')  # Start node
            elif node == end:
                node_colors.append('magenta')  # End node
            elif node in path:
                node_colors.append('red')  # Path nodes
            else:
                node_colors.append('#A9BCD0')  # Dull blue-gray for regular nodes

        # Draw the graph with updated colors
        nx.draw(G, positions,
                labels={i: node_names[i] for i in G.nodes},
                node_color=node_colors,
                edge_color='#D3D3D3',  # Dull light gray edges
                with_labels=True,
                node_size=500,
                font_size=8)

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_size=7)

        # Show heuristic values slightly below nodes
        heuristic_labels = {i: f"h={heuristic_values[i]}" for i in G.nodes}
        label_pos = {i: (positions[i][0], positions[i][1] - 3.5) for i in G.nodes}
        nx.draw_networkx_labels(G, label_pos, labels=heuristic_labels,
                                font_color='darkgreen', font_size=6)


        # Highlight the best path with strong color
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, positions,
                            edgelist=path_edges,
                            edge_color='crimson',
                            width=3)

        # Set graph bounds
        ax.set_xlim(min(x for x, y in positions.values()) - 10, max(x for x, y in positions.values()) + 10)
        ax.set_ylim(min(y for x, y in positions.values()) - 10, max(y for x, y in positions.values()) + 10)

        # Display total path cost at bottom
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        bottom_x = (x_min + x_max) / 2
        bottom_y = y_min + (y_max - y_min) * 0.02
        plt.text(bottom_x, bottom_y, f"Total Path Cost: {path_cost}",
                fontsize=12, ha='center', color='black')

        # Final render
        plt.title(f"Shortest Path from {start_name} to {end_name}", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Check loop condition
        ch = input('Search another path? (y/n): ').strip().lower()


if __name__ == '__main__':
    main()