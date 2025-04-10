import networkx as nx
import random
import numpy as np

# Modify jump_env() function for your structure. Don't change other functions.
def jump_env(structure, start_site, end_site, env_structure, lattice=None):
    """
    Determine the jump type and environment string for a jump between two sites.

    For the provided start_site and end_site, this function:
      - Determines the jump type based on the site labels.
      - Computes nearest neighbors for both start and end sites.
      - Finds common neighboring environments.
      - Constructs an environment string summarizing the neighbors.

    If an environment structure is not provided, it is computed using
    define_env_distribution.
    """

    # Define coordination numbers for each site type.
    coordination = {'Li-tet': 4, 'Li-oct': 6}

    # Retrieve the labels for the starting and ending sites.
    start_label = structure.labels[start_site]
    end_label = structure.labels[end_site]

    # Determine jump type from the labels (e.g., 'tet-tet', 'tet-oct', etc.)
    jump_type = start_label.split('-')[-1] + '-' + end_label.split('-')[-1]

    # Get the number of neighbors based on coordination numbers.
    num_neighbors_start = coordination[start_label]
    num_neighbors_end   = coordination[end_label]

    # Calculate the distance matrix between transition sites and environment sites.
    if lattice is None:
        lattice = structure.lattice
    dists = lattice.get_all_distances(structure.frac_coords, env_structure.frac_coords)

    # Find indices of the nearest neighbors for start and end sites.
    start_indices = np.argsort(dists[start_site, :])[:num_neighbors_start]
    end_indices   = np.argsort(dists[end_site, :])[:num_neighbors_end]

    # Build lists containing tuples (label, species symbol) for start and end neighbors.
    start_list = [(env_structure.labels[idx], env_structure.species[idx].symbol) for idx in start_indices]
    end_list   = [(env_structure.labels[idx], env_structure.species[idx].symbol) for idx in end_indices]

    # Determine common neighbors between start and end by checking intersection.
    end_set = set(end_indices)
    common_list = [(env_structure.labels[idx], env_structure.species[idx].symbol)
                   for idx in start_indices if idx in end_set]

    # Sort the lists lexicographically and join the symbols into strings.
    start_str  = ''.join(symbol for _, symbol in sorted(start_list))
    end_str    = ''.join(symbol for _, symbol in sorted(end_list))
    common_str = ''.join(symbol for _, symbol in sorted(common_list))

    return jump_type, f"{start_str}-{end_str}--{common_str}"

def add_periodic_images(structure_li):
    """
    For the given pymatgen structure, insert explicit periodic images for sites
    (selected by idx_li) that lie at the minimum fractional coordinate in each axis.

    Parameters:
        structure: A pymatgen Structure object.
        idx_li: List or array of site indices to consider.

    Returns:
        A tuple of three dictionaries (for x, y, z), each mapping the new site index
        (periodic image) to the original site index.
    """
    # Get the fractional coordinates (rounded) of the sites to be considered.
    f_coords = np.round(structure_li.frac_coords, 3)
    # Find the minimum fractional coordinate for each axis.
    mins = np.min(f_coords, axis=0)

    one_to_zero = {}

    # Loop over each axis (0: x, 1: y, 2: z)
    for ax in range(3):
        # Find indices in f_coords where the coordinate equals the minimum for this axis.
        matching_indices = np.where(f_coords[:, ax] == mins[ax])[0]
        # Create a translation vector along the current axis.
        translation = np.zeros(3)
        translation[ax] = 1

        # Start index for new sites is the current number of sites.
        start_idx = len(structure_li.sites)
        mapping = {}

        for i, original_idx in enumerate(matching_indices):
            new_idx = start_idx + i
            # Insert a new periodic image for the site.
            structure_li.insert(
                idx=new_idx,
                species=structure_li.species[original_idx],
                coords=translation + structure_li.frac_coords[original_idx],
                coords_are_cartesian=False,
                label=structure_li.labels[original_idx]
            )
            # Record the mapping from new site index to original site index.
            mapping[new_idx] = original_idx

        one_to_zero[ax] = mapping

    # Return the mapping for the x, y, and z axes as a tuple.
    return structure_li, one_to_zero
#%%
def structure_to_graph(structure_li, structure_env, env_e_act, max_e_act=1, dist_threshold=3):
    dists = structure_li.distance_matrix
    # Create a boolean mask for the upper triangular part only.
    upper_mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    # Combine with the distance condition.
    mask = upper_mask & (dists < dist_threshold) & (dists != 0)
    rows, cols = np.where(mask)

    G = nx.DiGraph()

    for i, site in enumerate(structure_li.sites):
        G.add_node(i, label=site.label)

    for start, stop in zip(rows, cols):
        jump_type_1, env_1 = jump_env(structure=structure_li, start_site=start, end_site=stop, env_structure=structure_env)
        jump_type_2, env_2 = jump_env(structure=structure_li, start_site=stop, end_site=start, env_structure=structure_env)
        try:
            e_act_1 = env_e_act[(jump_type_1, env_1)]
            e_act_2 = env_e_act[(jump_type_2, env_2)]
            e_act = max(random.choice(np.arange(e_act_1-0.04, e_act_1+0.04, 0.01)), random.choice(np.arange(e_act_2-0.04, e_act_2+0.04, 0.01)))
            if e_act <= max_e_act:
                G.add_edge(start, stop, e_act=e_act)
        except KeyError:
            print(f'KeyError: There is no activation energy for {jump_type_1, env_1} or {jump_type_2, env_2}')
            continue

    return G, len(rows)
#%%
def test_node_connectivity(G, node, target_node_list):
    """
    Check if 'node' is connected (by any path) to any node in target_node_list.
    """
    for target_node in target_node_list:
        if nx.has_path(G, node, target_node):
            return True
    return False


def test_node_connectivity_with_periodicity(G, node, target_node_list, one_to_zero,
                                             tracker=None, tracker_disconnected=None,
                                             zero_nodes_connected=None, disconnected_zero_nodes=None):
    """
    Recursively test connectivity considering periodic boundary conditions.

    If a connection is found via a periodic image (using one_to_zero mapping),
    update the trackers and return the connectivity status.
    """
    # Initialize lists if not provided.
    if tracker is None:
        tracker = []
    if tracker_disconnected is None:
        tracker_disconnected = []
    if zero_nodes_connected is None:
        zero_nodes_connected = []
    if disconnected_zero_nodes is None:
        disconnected_zero_nodes = []

    for target_node in target_node_list:
        if nx.has_path(G, node, target_node):
            # If the mapped node is already tracked or any connection exists,
            # mark the current node as connected.
            if one_to_zero[target_node] in tracker or zero_nodes_connected:
                tracker.append(node)
                zero_nodes_connected.extend(tracker)
                return True, list(set(zero_nodes_connected)), list(set(disconnected_zero_nodes))
            elif one_to_zero[target_node] in disconnected_zero_nodes:
                continue
            else:
                tracker.append(node)
                tracker_disconnected.append(node)
                new_node = one_to_zero[target_node]
                return test_node_connectivity_with_periodicity(G, new_node, target_node_list, one_to_zero,
                                                                tracker, tracker_disconnected,
                                                                zero_nodes_connected, disconnected_zero_nodes)
    disconnected_zero_nodes.extend(tracker_disconnected + [node])
    return False, list(set(zero_nodes_connected)), list(set(disconnected_zero_nodes))


def test_connectitivity_of_node_list(G, node_list, target_node_list, one_to_zero,
                                     zero_nodes_connected=None, disconnected_zero_nodes=None):
    """
    For each node in node_list, test its connectivity to target_node_list.

    Returns lists of nodes that are connected (zero_nodes_connected)
    and those that are disconnected.
    """
    if zero_nodes_connected is None:
        zero_nodes_connected = []
    if disconnected_zero_nodes is None:
        disconnected_zero_nodes = []

    for node in node_list:
        if node in zero_nodes_connected:
            continue
        elif test_node_connectivity(G, node, zero_nodes_connected):
            zero_nodes_connected.append(node)
        else:
            flag, zero_nodes_connected, disconnected_zero_nodes = test_node_connectivity_with_periodicity(
                G, node, target_node_list, one_to_zero,
                zero_nodes_connected=zero_nodes_connected, disconnected_zero_nodes=disconnected_zero_nodes)
            for dis_node in disconnected_zero_nodes:
                if test_node_connectivity(G, dis_node, zero_nodes_connected):
                    zero_nodes_connected.append(dis_node)
                    if dis_node in disconnected_zero_nodes:
                        disconnected_zero_nodes.remove(dis_node)
    return list(set(zero_nodes_connected)), list(set(disconnected_zero_nodes))


def analyse_percolation_through_one_supercell(structure_li, structure_env, env_e_act, target_ea, one_to_zero, iterations=50):
    """
    For a single simulation of a supercell, build a graph with nodes from idxs_li.

    For each allowed jump, add an edge if the (randomized) jump activation energy
    is below the target. Then test connectivity between boundary nodes (using one_to_zero)
    and bulk nodes.

    Returns the fraction of connected sites and the probability of connection.
    """
    x_one_to_zero, y_one_to_zero, z_one_to_zero = (one_to_zero[0],one_to_zero[1],one_to_zero[2])
    fraction_connected_sites_list = []
    probability_of_connection_list = []

    for simulation in range(iterations):
        # Build the graph.
        G, num_possible_jumps = structure_to_graph(structure_li, structure_env, env_e_act, max_e_act=target_ea, dist_threshold=3)

        # Test connectivity for each boundary direction.
        results_x = test_connectitivity_of_node_list(G,
                                                     list(x_one_to_zero.values()),
                                                     list(x_one_to_zero.keys()),
                                                     x_one_to_zero)
        results_y = test_connectitivity_of_node_list(G,
                                                     list(y_one_to_zero.values()),
                                                     list(y_one_to_zero.keys()),
                                                     y_one_to_zero)
        results_z = test_connectitivity_of_node_list(G,
                                                     list(z_one_to_zero.values()),
                                                     list(z_one_to_zero.keys()),
                                                     z_one_to_zero)

        # Determine bulk nodes (those not on any boundary).
        total_boundary_nodes = np.array(list(x_one_to_zero.values()) + list(y_one_to_zero.values()) +
                                        list(z_one_to_zero.values()) + list(x_one_to_zero.keys()) +
                                        list(y_one_to_zero.keys()) + list(z_one_to_zero.keys()))
        bulk_nodes = [element for element in list(range(len(structure_li.sites))) if element not in total_boundary_nodes]

        # Start with the union of all connected boundary nodes.
        connected_nodes_total = list(set(results_x[0] + results_y[0] + results_z[0]))
        for bulk_node in bulk_nodes:
            if test_node_connectivity(G, bulk_node, connected_nodes_total):
                connected_nodes_total.append(bulk_node)

        fraction_connected_sites_list.append(len(set(connected_nodes_total)) / len(structure_li.sites))
        probability_of_connection_list.append(len(G.edges()) / num_possible_jumps)

    return fraction_connected_sites_list, probability_of_connection_list


def test_energy(ea, structure_li, structure_env, env_e_act, one_to_zero, iterations_graph=50, iterations_r_structure=50):
    """
    For a given target energy (ea), perform several simulations (without parallelization)
    to analyze percolation through one supercell.

    For each simulation, the graph is built and percolation is analyzed. The function
    returns aggregated fractions and connection probabilities and writes the results to a file.
    """
    fractions = []
    probs = []
    probs_outer = {}
    fractions_outer = {}

    for random_simulation in range(iterations_r_structure):
        structure_env = make_random_structure()
        f, p = analyse_percolation_through_one_supercell(structure_li, structure_env, env_e_act, ea, one_to_zero, iterations=iterations_graph)

        fractions.extend(f)
        probs.extend(p)

    a = np.array(fractions)
    a = a[a > 0]
    std_probs = np.std(probs)
    std_a = np.std(a)

    probs_outer[ea] = [np.average(probs), std_probs, std_probs / np.sqrt(len(probs))]
    fractions_outer[ea] = [np.average(a), std_a, std_a / np.sqrt(len(a))]

    tmp_dict = {'fractions': fractions_outer,
                'probs': probs_outer}

    return tmp_dict