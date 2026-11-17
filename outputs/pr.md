Default mtkcompile throws a BoundsError for stationary systems with an empty equation row when every variable is matched.
The inverse matching in large_linear_sccs is indexed by equation ID but was sized by the number of variables.
Size it with nsrcs(graph) so every equation row has an entry.
Add a minimal regression test that compiles the three-equation system and checks that one unknown remains.
