package graph

var _ Grapher = new(GraphVizGrapher)

// GraphVizGrapher ...
type GraphVizGrapher struct {
	nodes []*Node
	edges []*Edge
}

func (_ *GraphVizGrapher) Render() error {
	return nil
}

func (_ *GraphVizGrapher) AddNode(n *Node) error       { return nil }
func (_ *GraphVizGrapher) AddEdge(n, m *Node, e *Edge) {}
