from trex import

plan = EnginePlan(
    "my-engine.graph.json",
    "my-engine.profile.json",
    "my-engine.profile.metadata.json")

graph = to_dot(plan, layer_type_formatter)
svg_name = render_dot(graph, engine_name, 'svg')


inspector = engine.create_engine_inspector();
inspector.execution_context = context; # OPTIONAL
print(inspector.get_layer_information(0, LayerInformationFormat.JSON); # Print the information of the first layer in the engine.
print(inspector.get_engine_information(LayerInformationFormat.JSON); # Print the information of the entire engine.