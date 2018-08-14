import numpy as np
import pandas as pd

def parse_data(data, CCM_thresh, weighted=False):

    '''
    Parses CCM dataframe and returns edgelist

    Args:
    -----
    data [dataframe]: Dataframe with shape
                       nodeA nodeB ccm_coeff_AB ccm_coeff_BA

    ccm_thresh [float]: only edges with weights >= thresh
                        will be returned

    weighted [bool]: returns binary edges when False,
                      returns weighted edges when True

    Returns:
    ---------

    edgelist [list] : list of tuple edges with the shape
                        (source, target, weight)

    '''

    if weighted:
        xy = data[(data['p_xy'] <= 0.05) & (data['pearson_xy']>= CCM_thresh)]
        xy.set_index([list(range(len(xy)))], inplace=True)
        edgelist_cxy = [(xy.src[i], xy.tgt[i], xy.pearson_xy[i]) for i in range(len(xy))]

        yx = data[(data['p_yx'] <= 0.05) & (data['pearson_yx']>= CCM_thresh)]
        yx.set_index([list(range(len(yx)))], inplace=True)
        edgelist_cyx = [(yx.tgt[i], yx.src[i], yx.pearson_yx[i]) for i in range(len(yx))]

        edgelist = edgelist_cxy+edgelist_cyx

    else:
        xy = data[(data['p_xy'] <= 0.05) & (data['pearson_xy']>= CCM_thresh)]
        xy.set_index([list(range(len(xy)))], inplace=True)
        edgelist_cxy = [(xy.src[i], xy.tgt[i]) for i in range(len(xy))]

        yx = data[(data['p_yx'] <= 0.05) & (data['pearson_yx']>= CCM_thresh)]
        yx.set_index([list(range(len(yx)))], inplace=True)
        edgelist_cyx = [(yx.tgt[i], yx.src[i]) for i in range(len(yx))]

        edgelist = edgelist_cxy+edgelist_cyx

    return edgelist


def weighted_adj_matrix(edgelist):

    origin=[i[0] for i in edgelist]
    destination=[i[1] for i in edgelist]
    weights=[float(i[2]) for i in edgelist]
    countries = list(set(origin+destination))

    # because it doesn't work if our nodes are strings
    node_mapping = {countries[i]:i for i in range(len(countries))}
    origin_recoded = [node_mapping[i] for i in origin]
    destination_recoded = [node_mapping[i] for i in destination]

    # create matrix
    adjMatrix = [[0 for i in range(len(countries))] for k in range(len(countries))]

    # scan the arrays edge_u and edge_v
    for i in range(len(origin_recoded)):
        u = origin_recoded[i]
        v = destination_recoded[i]
        adjMatrix[u][v] = weights[i]

    adjMatrix = np.array(adjMatrix)

    return adjMatrix, node_mapping


def directed_weighted_rc(M, country_indices, members=False):

    """
    Returns the node strength between nodes of strength k and higher.

    Parameters:
    -----------
    A [np.array] : adjacecy matrix

    Returns:
    ---------
    phis_num [list]: sum of the strength of connections between nodes of strength k >

    """

    degree_dist = np.sum(M, axis=1)
    ranks = sorted(set(degree_dist), reverse=False)

    phis_num = []
    number_nodes = []
    cumulative_strengths = []
    mems = []

    ordered_members = np.array([country_indices[key] for key in sorted(country_indices.keys())])

    for k in range(len(ranks)):

        # get indices of nodes with strength < k
        small_nodes_indices = np.where(degree_dist < ranks[k])

        # Delete nodes
        A_trimmed = np.delete(np.delete(M,small_nodes_indices,axis=0), small_nodes_indices, axis=1)

        ordered_mems = np.delete(ordered_members, small_nodes_indices)

        # Make list of nodes still in the net
        mems.append(list(ordered_mems))

        # Make list of number of nodes with srength >= k
        number_nodes.append(len(A_trimmed))

        # Compute 'amount of influence' exerted by nodes
        cumulative_strengths.append(np.sum(A_trimmed, axis=1))

        # total weight of connections amongst nodes with highest out-strength
        total_strength = np.sum(A_trimmed)
        phis_num.append(total_strength)

    if members:
        return phis_num,number_nodes, cumulative_strengths, mems
    else:
        return phis_num,number_nodes, cumulative_strengths

def shuffle_weighted_links(edgelist,swaps=1000, weighted=True):

    """
    Shuffles the links in a directed weighted network. It keeps
    the out-degree constant and the weights attached to the weights.
    Hence, in the end, the out-strength is preserved.

    Parameters:
    -----------
    edgelist [list of tuples]: network edgelist with weights (A,B, weight)
    Swaps [int]: numper of swaps to perform in the network

    Returns:
    ---------
    edgelist [list of tuples] = shuffled edge list

    """

    # pick two edges at random
    randints_1 = list(np.random.randint(len(edgelist), size=swaps))
    randints_2 = list(np.random.randint(len(edgelist), size=swaps))

    for i in range(swaps):
        edge1 = edgelist[randints_1[-1]]
        edge2 = edgelist[randints_2[-1]]

        # if edge doesnt exist already, create it
        if (edge1[0], edge2[1]) not in edgelist and (edge2[0], edge1[1]) not in edgelist:
            if weighted:
                edgelist[randints_1[-1]] = (edge1[0],edge2[1], edge1[2])
                edgelist[randints_2[-1]] = (edge2[0],edge1[1], edge2[2])
            else:
                edgelist[randints_1[-1]] = (edge1[0],edge2[1])
                edgelist[randints_2[-1]] = (edge2[0],edge1[1])

        randints_1.pop()
        randints_2.pop()

    return edgelist


def dw_rich_club_info(edges, return_country_strengths=False):
    '''
    Args:
    -----

    edges [list of tups] : list of edges (source, target, weight)
    '''

    # make weighted adj matrix
    A, node_mapping = weighted_adj_matrix(edges)

    # make index:country mapping from node_mapping
    country_indices = {v:k for k,v in node_mapping.items()}

    # Compute node out strength
    strengths = list(np.sum(A, axis=1))

    # Make country:strength pairing - list because it is ordered
    country_strengths = [(country_indices[node], strengths[node]) for node in range(len(strengths))]

    # -----  Get rich club stregth from observed network ----- #

    phis_num, number_of_nodes,cum_strengths,mems = directed_weighted_rc(A,country_indices, members=True)


    # -----  Get rich club stregth from simulations   ----- #

    random_phis_array = np.zeros((1000, len(phis_num)))
    random_number_of_nodes = np.zeros((1000, len(number_of_nodes)))

    for i in range(1000):
        rand=shuffle_weighted_links(edges)
        A_rand, b=weighted_adj_matrix(rand)
        random_phis_array[i],n_nodes,h=directed_weighted_rc(A_rand, country_indices, members=False)
        random_number_of_nodes[i]=n_nodes

    phis_den = np.mean(random_phis_array,axis=0)
    mean_random_nodes = np.mean(random_number_of_nodes,axis=0)
    phis=[phis_num[i]/phis_den[i] for i in range(len(phis_num))]

    # zscores
    zscores = []
    for i in range(len(phis_num)):
        sigma = np.std(random_phis_array[:,i])
        mean = np.mean(random_phis_array[:,i])
        zscores.append(abs(phis_num[i]-mean)/sigma)

    results=pd.DataFrame()
    results['phis_rhos'] = phis
    results['z_scores'] = zscores
    results['number_of_nodes'] = number_of_nodes
    results['rand_mean_num_of_nodes'] = mean_random_nodes
    results['observed_strengths'] = phis_num
    results['rand_strengths'] = phis_den
    results['members'] = mems

    if return_country_strengths:
        strengths=pd.DataFrame()
        strengths['country']=[i[0] for i in country_strength]
        strengths['strength']=[i[1] for i in country_strength]

        return strengths

    else:

        return results
