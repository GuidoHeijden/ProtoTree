from util.log import Log


def measure_dist(tree, original_tree, node_ixs=None):
    log = Log("./replacements/logs/distance_measurements")
    if node_ixs == None:
        node_ixs = [node.index for node in tree.nodes]

    log.log_message("\nClever Hans nodes per depth:")
    nodes_at_depth_i = {}
    for cur_depth in range(1, tree.depth):
        nodes_at_depth_i[cur_depth] = [node_ix for node_ix in node_ixs if tree.nodes_by_index[node_ix].depth == cur_depth]
        log.log_message("\n"+str(cur_depth)+": "+str(nodes_at_depth_i[cur_depth]))

        for node_ix in nodes_at_depth_i[cur_depth]:
            a = original_tree.prototype_layer.prototype_vectors[original_tree._out_map[original_tree.nodes_by_index[node_ix]]]
            b = tree.prototype_layer.prototype_vectors[tree._out_map[tree.nodes_by_index[node_ix]]]
            dist = sum(((a-b)**2).reshape(tree.prototype_shape[2]))  # Euclidean distance
            log.log_message("Distance original and replacement for node "+str(node_ix)+": "+str(dist))
