from tqdm import tqdm

"""This function is designed to create nodes in a Neo4j graph database without establishing any relationships between them.
 It takes a graph object, a dictionary of data, and strings for the main node’s label and name. First, it creates (or merges) a 
 main node using the provided label and name, ensuring that the node is only created if it doesn't already exist. Then, it iterates 
 through the keys in the data dictionary—each representing a section—and for each one, it creates a Section node with 
 two properties: type, which is the name of the section, and parent_name, which references the main node's name. 
 This setup adds all the nodes to the database but does not yet define any relationships between them, making it useful 
 for initial node population before linking them.
"""


# 1. Add main nodes without creating relationships
def create_nodes(graph, data: dict, node_label: str, node_name: str):
    # Create the main node
    main_node_query = f"""
    MERGE (main:{node_label} {{name: $name}})
    """
    graph.query(main_node_query, params={"name": node_name})

    # Create section nodes only (without relationships)
    for section, content in data.items():
        query = f"""
        MERGE (s:Section {{type: $type, parent_name: $name}})
     
        """
        params = {
            "type": section,
            "name": node_name
        }
        graph.query(query, params=params)

