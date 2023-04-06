package graph

import "github.com/shopspring/decimal"

var grapher Grapher

func Init(g Grapher) {
	grapher = g
}

type Node struct {
	Data    decimal.Decimal
	Grad    decimal.Decimal
	Operand string
	ID      string
}

type Edge struct {
	Weight decimal.Decimal
	Bias   decimal.Decimal
}

type Grapher interface {
	Render() error
	AddNode(n *Node) error
	AddEdge(n, m *Node, e *Edge)
}

func Render() error {
	if grapher != nil {
		return grapher.Render()
	}

	return nil
}

func AddNode(n *Node) error {
	if grapher != nil {
		return grapher.AddNode(n)
	}

	return nil
}

func AddEdge(n, m *Node, e *Edge) {
	if grapher != nil {
		grapher.AddEdge(n, m, e)
	}
}
