"""Cases where the result is deliberately not the original's, and why.

``tests/test_golden.py`` skips these.  ``tests/test_intentional_changes.py`` checks the new behaviour instead.
"""

DEFECT = "the original failed because of a defect (an undefined name, a wrong keyword, an ambiguous comparison)"
WIDER = "an input that the original refused for no good reason is accepted now"
PANDAS = "the original gave a pandas-version dependent outcome; this is an explicit error now"
HYGIENE = "package exports or import side effects that were accidents of the original"

CHANGES: dict[str, str] = {
    # ---- local
    "local.neighborhood_of_set.default_include_centers": DEFECT + ": nodesA",
    "local.neighborhood_of_set.include_centers_return_neighbors": DEFECT + ": nodesA",
    "local.neighborhood_indices.duplicate_centers": PANDAS + " (TypeError on pandas 2)",
    "local.neighborhood_indices.duplicate_isolated_centers_both": PANDAS,
    "local.neighborhood_indices.duplicate_isolated_centers_pre_only": PANDAS
    + " (TypeError on pandas 2)",
    "local.neighborhood_of_set_indices.repeated_nodes": PANDAS + " (TypeError on pandas 2)",
    # ---- modelling
    "modelling.conn_prob_2nd_order_model.coo_adjacency_cannot_be_indexed_without_subsampling": WIDER
    + ": COO",
    "modelling.extract_dependent_p_conn.format_coo": WIDER + ": COO",
    "modelling.run_batch_model_building.adjacency_saved_as_coo_without_subsampling": WIDER
    + ": COO",
    "modelling.run_model_building.order2_coo_adjacency_without_subsampling": WIDER + ": COO",
    "modelling.run_pathway_model_building.order2_coo_adjacency_without_subsampling": WIDER
    + ": COO",
    "modelling.import.logging_is_configured_for_the_whole_process": HYGIENE
    + ": logging.basicConfig at import",
    # ---- randomization
    "randomization.add_rc_connections_skeleta.dimensions_as_array_fails_while_computing_skeleta": DEFECT
    + ": `== None`",
    "randomization.run_DD2_model.fit_requires_a_broken_import": DEFECT
    + ": `from .import modelling` found no module",
    "randomization.exports.rand_utils_callables": HYGIENE
    + ": typing.List and typing.Union are no longer imported",
    "randomization.exports.package_callables": HYGIENE + ": KDTree is no longer re-exported",
    "randomization.exports.top_level_package_all": HYGIENE
    + ": __all__ of connalysis listed a module that does not exist",
    "randomization.exports.top_level_star_import_fails_on_rand_utils": HYGIENE
    + ": from connalysis import * works",
    "classic.closeness_connected_components.explicit_zero_bridge": DEFECT
    + ": ValueError from scikit-network, no longer used",
    "classic.common_neighbor_connectivity_bias.bad_direction": DEFECT
    + ": UnboundLocalError, a ValueError now",
    "classic.common_neighbor_weight_bias.bad_direction": DEFECT
    + ": UnboundLocalError, a ValueError now",
    "classic.analytical_expected_gini_curve.bad_direction": DEFECT
    + ": UnboundLocalError, a ValueError now",
    "classic.connection_probability_within.single_frame_with_tiny_distance": DEFECT
    + ": ZeroDivisionError (or a result, "
    "depending on scipy) for the mean of no pairs; it is NaN now, as when no pair is within reach",
    "classic.distribution_number_of_common_neighbors.dense_input": WIDER + ": dense array",
    "classic.normalized_rich_club_curve.afferent": DEFECT
    + ": direction names that no function accepted",
    "classic.normalized_rich_club_curve.analytical_null_model": DEFECT
    + ": direction names that no function accepted",
    "classic.normalized_rich_club_curve.default_arguments": DEFECT
    + ": direction names that no function accepted",
    "classic.normalized_rich_club_curve.mean_normalisation": DEFECT
    + ": direction names that no function accepted",
    "classic.rich_club_coefficient.analytical_null_model": DEFECT
    + ": direction names that no function accepted",
    "classic.rich_club_coefficient.default_arguments": DEFECT
    + ": direction names that no function accepted",
    # ---- topology
    "topology.euler_characteristic.undefined_name_in_the_function_body": DEFECT + ": NameError",
    "topology.simplicial_rich_club_curve.maximal_simplices": DEFECT + ": wrong keyword",
    "topology.simplicial_rich_club_curve.sparse_bin_set": DEFECT + ": wrong keyword",
    "topology.simplicial_rich_club_curve.wrong_keyword": DEFECT + ": wrong keyword",
    "topology.tcc.center_argument": DEFECT + ": reciprocal_connections did not exist",
    "topology.tcc.imports_a_name_that_does_not_exist": DEFECT
    + ": reciprocal_connections did not exist",
    "topology.bedge_counts.dense_adjacency": WIDER + ": dense array",
    "topology.count_triads_fully_connected.dense_array_is_not_accepted": WIDER + ": dense array",
    "topology.node_degree.np_matrix_input": WIDER + ": numpy matrix",
    "topology.get_k_skeleta_graph.dimensions_numpy_array": DEFECT + ": `== None`",
    "topology.get_k_skeleta_graph.dimensions_numpy_array_spine": DEFECT + ": `== None`",
    "topology.betti_counts.approximation_numpy_array": DEFECT
    + ": `== None`; still rejected, now with an AssertionError",
}


# Results that depend on the version of numpy: numpy 1 promoted the dtype of float32 scalars differently (NEP 50)
NUMPY_1_DTYPES = {
    "modelling.get_model_function.second_order_scalar_type",
    "stats.node_stats_per_position_single.value_dtypes",
}
