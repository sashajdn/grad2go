package graph

import (
	"bytes"
	"fmt"

	"github.com/shopspring/decimal"
)

var grapher Grapher

type NodeKind int32

const (
	NodeKindInput NodeKind = iota + 1
	NodeKindBias
	NodeKindWeight
	NodeKindValue
	NodeKindOperator
)

func Init(g Grapher) {
	if g != nil {
		grapher = g
	}
}

// TODO: pass label.
func NewNode(data, grad float64, operand, id string, kind NodeKind) *Node {
	d, g := decimal.NewFromFloat(data), decimal.NewFromFloat(grad)

	return &Node{
		Data:    d,
		Grad:    g,
		Operand: operand,
		ID:      id,
		Kind:    kind,
	}
}

func NewNodeWithLayer(data, grad float64, operand, id, layer string, kind NodeKind) *Node {
    n := NewNode(data, grad, operand, id, kind)
    n.Layer = layer
    return n
}

type Node struct {
	Data    decimal.Decimal
	Grad    decimal.Decimal
	Kind    NodeKind
	Operand string
	ID      string
	Label   string
	Layer   string
}

type Edge struct {
	V, U   *Node
	Weight decimal.Decimal
	Bias   decimal.Decimal
	ID     string
}

type Grapher interface {
	ResetGraph() error
	Render() (*bytes.Buffer, error)
	AddNode(n *Node) error
	AddEdge(n, m *Node, e *Edge) error
}

func Render() (*bytes.Buffer, error) {
	if grapher != nil {
		return grapher.Render()
	}

	return nil, fmt.Errorf("grapher not initialised")
}

func ResetGraph() error {
	if grapher != nil {
		return grapher.ResetGraph()
	}

	return fmt.Errorf("grapher not initialised")
}

func AddNode(n *Node) error {
	if grapher != nil {
		return grapher.AddNode(n)
	}

	return fmt.Errorf("grapher not initialised")
}

func AddEdge(n, m *Node, e *Edge) error {
	if grapher != nil {
		return grapher.AddEdge(n, m, e)
	}

	return fmt.Errorf("grapher not initialised")
}
