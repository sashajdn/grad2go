package nn

import (
	"fmt"
	"grad2go/graph"
	"strconv"
	"strings"
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
		// Add original value as node.
		if _, ok := setOfGraphNodes[node.ID()]; ok {
			return
		}
		setOfGraphNodes[node.ID()] = marshalValueToNode(node)

		var workingNode = marshalValueToNode(node)
		if node.operation.String() != "noop" {
			// Create "operand" node & inject.
			cp := deepCopy(workingNode)
			cp.Kind = graph.NodeKindOperator
			cp.ID = cp.ID + cp.Operand

			setOfGraphNodes[cp.ID] = cp

			// Create edge between original value & operand.
			eID := edgeID{workingNode.ID, cp.ID}
			setOfGraphEdges[eID] = [2]*graph.Node{workingNode, cp}

			// We now want all the child node values to be "inputs" to the operand node.
			workingNode = cp
		}

		// Create an edges.
		for _, child := range node.previous {
			eID := edgeID{node.ID(), child.ID()}

			if _, ok := setOfGraphEdges[eID]; !ok {
				v := workingNode
				u := marshalValueToNode(child)

				setOfGraphEdges[eID] = [2]*graph.Node{v, u}
			}

			build(child)
		}
	}
	build(root)

	// Build nodes.
	for _, node := range setOfGraphNodes {
		if err := g.AddNode(node); err != nil {
			return fmt.Errorf("add node %s: %w", node.ID, err)
		}
	}

	// Build edges.
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

func BuildGraphVizLayersString(layers int) string {
	var intermediary = make([]string, layers)
	for i := 0; i < layers; i++ {
		intermediary[i] = strconv.Itoa(i)
	}

	return strings.Join(intermediary, ":")
}

func marshalValueToNode(v *Value) *graph.Node {
	var kind graph.NodeKind
	switch v.Kind() {
	case KindBias:
		kind = graph.NodeKindBias
	case KindWeight:
		kind = graph.NodeKindWeight
	case KindInput:
		kind = graph.NodeKindInput
	case KindValue:
		kind = graph.NodeKindValue
	}

	var layer string
	if layerValue := v.layer(); layerValue >= 0 {
		layer = strconv.Itoa(layerValue)
	}

	return &graph.Node{
		Data:    v.data,
		Grad:    v.grad,
		Operand: v.operation.String(),
		ID:      v.ID(),
		Label:   v.Label(),
		Kind:    kind,
		Layer:   layer,
	}
}

func deepCopy(n *graph.Node) *graph.Node {
	d, _ := n.Data.Float64()
	g, _ := n.Grad.Float64()

	return graph.NewNode(d, g, n.Operand, n.ID, n.Kind)
}
