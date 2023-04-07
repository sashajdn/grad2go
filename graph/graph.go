package graph

import (
	"bytes"
	"fmt"

	"github.com/shopspring/decimal"
)

var grapher Grapher

func Init(g Grapher) {
	if g != nil {
		grapher = g
	}
}

type Node struct {
	Data    decimal.Decimal
	Grad    decimal.Decimal
    IsOperandNode bool
	Operand string
	ID      string
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