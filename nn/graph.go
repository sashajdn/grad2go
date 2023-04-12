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
		if err := g.AddNode(node); err != nil {
			return fmt.Errorf("add node %s: %w", node.ID, err)
		}

		// Add "faked" operand node for visibility.
		if node.Operand != "noop" {
			cp := deepCopy(node)
			cp.IsOperandNode = true
			cp.ID = cp.ID + cp.Operand

			if err := g.AddNode(cp); err != nil {
				return fmt.Errorf("add node copy as operand node %s: %w", cp.ID, err)
			}

			edgeID := fmt.Sprintf("%s:%s", cp.ID, node.ID)

			if err := g.AddEdge(cp, node, &graph.Edge{
				ID: edgeID,
			}); err != nil {
				return fmt.Errorf("add edge %s: %w", edgeID, err)
			}

			setOfGraphNodes[node.ID] = cp
		}
	}

	for _, edge := range setOfGraphEdges {
		v, u := edge[0], edge[1]

		edgeID := fmt.Sprintf("%s:%s", v.ID, u.ID)
		if err := g.AddEdge(u, v, &graph.Edge{
			ID: edgeID,
		}); err != nil {
			return fmt.Errorf("add edge %s: %w", edgeID, err)
		}
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
		Label:         v.Label(),
	}
}

func deepCopy(n *graph.Node) *graph.Node {
	d, _ := n.Data.Float64()
	g, _ := n.Grad.Float64()

	return graph.NewNode(d, g, n.Operand, n.ID, n.IsOperandNode)
}
