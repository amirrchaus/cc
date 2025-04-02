def page_rank(graph: dict[str, list[str]], k: int, d: float = 0.85) -> dict[str, float]:
    """
    Calculate the PageRank of nodes in a directed graph.
    
    Parameters:
        graph (dict): A dictionary representing the graph where keys are nodes 
                      and values are lists of nodes they point to.
        k (int): Number of iterations to run the PageRank algorithm.
        d (float): Damping factor (default is 0.85).
    
    Returns:
        dict: A dictionary with nodes as keys and their PageRank scores as values.
    """
    nodes = graph.keys()
    rank = {node: 1.0 for node in nodes}  # Initialize all ranks to 1.0
    
    for _ in range(k):
        new_rank = {}
        for node in nodes:
            incoming_sum = sum(
                (new_rank[subnode] if subnode in new_rank else rank[subnode]) / len(graph[subnode])
                for subnode, outnodes in graph.items()
                if node in outnodes and len(graph[subnode]) > 0
            )
            new_rank[node] = (1 - d) + d * incoming_sum
        rank = new_rank
    return rank

if __name__ == "__main__":
    graph = {}
    num_nodes = int(input("Enter the number of nodes in the graph: "))
    
    for _ in range(num_nodes):
        node = input("Enter node name: ")
        edges = input(f"Enter the nodes {node} points to (comma-separated, leave empty if none): ").split(',')
        graph[node] = [edge.strip() for edge in edges if edge.strip()]  # Remove empty strings
    
    iterations = int(input("Enter the number of iterations: "))
    ranks = page_rank(graph, iterations)
    
    for node, rank in ranks.items():
        print(f"Rank of {node} is {rank}")
