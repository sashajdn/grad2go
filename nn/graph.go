package nn

import (
	"fmt"
	"grad2go/graph"
)

func BuildGraphFromRootValue(g graph.Grapher, root *Value) error {
	if g == nil {
		return fmt.Errorf("graph nil; cannot build with an empty graph")
	}

	// Reset the graph.
	if err := g.ResetGraph(); err != nil {
		return fmt.Errorf("failed to reset graph: %w", err)
	}

	type edgeID struct {
		v, u string
	}

	setOfGraphNodes := make(map[string]*graph.Node)
	setOfGraphEdges := make(map[edgeID][2]*graph.Node)

	// DFS.
	var build func(node *Value)
	build = func(node *Value) {
		if _, ok := setOfGraphNodes[node.ID()]; ok {
			return
		}
		setOfGraphNodes[node.ID()] = marshalValueToNode(node)

		for _, child := range node.previous {
			eID := edgeID{node.ID(), child.ID()}

			if _, ok := setOfGraphEdges[eID]; !ok {
				v := marshalValueToNode(node)
				u := marshalValueToNode(child)

				setOfGraphEdges[eID] = [2]*graph.Node{v, u}
			}

			build(child)
		}
	}
	build(root)

	for _, node := range setOfGraphNodes {
		g.AddNode(node)

		// Add "faked" operand node for visibility.
		if node.Operand != "noop" {
			cp := node
			cp.IsOperandNode = true
			cp.ID = cp.ID + cp.Operand

			g.AddNode(cp)
			g.AddEdge(cp, node, &graph.Edge{
				ID: fmt.Sprintf("%s:%s", cp.ID, node.ID),
			})
		}
	}

	for _, edge := range setOfGraphEdges {
		v, u := edge[0], edge[1]
		g.AddEdge(v, u, &graph.Edge{
			ID: fmt.Sprint("%s:%s", v.ID, u.ID),
		})
	}

	return nil
}

func marshalValueToNode(v *Value) *graph.Node {
	return &graph.Node{
		Data:          v.data,
		Grad:          v.grad,
		Operand:       v.operation.String(),
		ID:            v.ID(),
		IsOperandNode: false,
	}
}
