"""Github ontology contribution (CONCEPT:AU-KG.ontology.federation-provider-leg).

Data-only subpackage: it carries ``github.ttl`` (the ``owl:Ontology``
``http://knuckles.team/kg/github`` module — organizations, repositories, pull requests,
issues, releases, branches, commits, workflows and workflow runs with their review and
delivery relationships) which the scion-sdk hub federates in via
the ``scion_sdk.ontology_providers`` entry-point. It holds no business logic
and no heavy imports so the hub can resolve it cheaply.
"""
