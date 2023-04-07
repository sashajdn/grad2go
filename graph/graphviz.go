package graph

import (
	"bytes"
	"fmt"
	"sync"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

var (
	// TODO: enumerate
	defaultNodeColor = "white"
	defaultEdgeColor = "red"
)

func NewGraphVizGraph(cfg GraphVizConfig) (*GraphVizGraph, error) {
	gv := graphviz.New()
	g, err := gv.Graph(cfg.GraphVizOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create new graph: %w", err)
	}

	return &GraphVizGraph{
		cfg:    cfg,
		cnodes: make(map[string]*cgraph.Node),
		nodes:  make([]*Node, 0),
		edges:  make([]*Edge, 0),
		g:      graphviz.New(),
		cg:     g,
	}, nil
}

var _ Grapher = new(GraphVizGraph)

// GraphVizGraph ...
type GraphVizGraph struct {
	// Config.
	cfg GraphVizConfig

	// Caches.
	cnodes  map[string]*cgraph.Node
	nodes   []*Node
	nodesMu sync.RWMutex
	edges   []*Edge
	cedges  map[string]*cgraph.Edge

	// Graph.
	g    *graphviz.Graphviz
	cg   *cgraph.Graph
	cgMu sync.RWMutex
}

func (g *GraphVizGraph) ResetGraph() error {
	cg, err := g.g.Graph(g.cfg.GraphVizOpts...)
	if err != nil {
		return fmt.Errorf("failed to build new cgraph: %w", err)
	}

	// Reset internal caches.
	g.nodesMu.Lock()
	g.nodes = make([]*Node, 0)
	g.cnodes = make(map[string]*cgraph.Node)

	g.edges = make([]*Edge, 0)
	g.cedges = make(map[string]*cgraph.Edge)
	g.nodesMu.Unlock()

	g.cgMu.Lock()
	defer g.cgMu.Unlock()

	g.cg = cg
	return nil
}

func (g *GraphVizGraph) Render() (*bytes.Buffer, error) {
	var buf = &bytes.Buffer{}
	if err := g.g.Render(g.cg, graphviz.SVG, buf); err != nil {
		return nil, fmt.Errorf("failed to render graph to buffer: %w", err)
	}

	return buf, nil
}

func (g *GraphVizGraph) AddNode(n *Node) error {
	cnode, err := g.cg.CreateNode(n.ID)
	if err != nil {
		return fmt.Errorf("failed to create graph node: %w", err)
	}

	label := buildLabelFromNode(n)
	cnode.SetLabel(label)
	cnode.SetColor(stringOrDefault(g.cfg.NodeColor, defaultNodeColor))

	g.setNode(n.ID, cnode)
	g.nodes = append(g.nodes, n)

	return nil
}

func (g *GraphVizGraph) AddEdge(n, m *Node, e *Edge) error {
	cn, ok := g.readNode(n.ID)
	if !ok {
		return fmt.Errorf("node `n=%s` cannot be found", n.ID)
	}

	cm, ok := g.readNode(m.ID)
	if !ok {
		return fmt.Errorf("node `m=%s` cannot be found", m.ID)
	}

	cedge, err := g.cg.CreateEdge(e.ID, cn, cm)
	if err != nil {
		return fmt.Errorf("failed to create graph edge: %w", err)
	}

	label := buildLabelFromEdge(e)
	cedge.SetLabel(label)
	cedge.SetColor(stringOrDefault(g.cfg.EdgeColor, defaultEdgeColor))

	g.cedges[e.ID] = cedge // TODO: take lock
	g.edges = append(g.edges, e)

	return nil
}

func (g *GraphVizGraph) setNode(id string, n *cgraph.Node) {
	g.nodesMu.Lock()
	defer g.nodesMu.Unlock()

	_, ok := g.cnodes[id]
	if ok {
		return
	}

	g.cnodes[id] = n
}

func (g *GraphVizGraph) readNode(id string) (*cgraph.Node, bool) {
	g.nodesMu.RLock()
	defer g.nodesMu.RUnlock()

	if n, ok := g.cnodes[id]; ok {
		return n, true
	}

	return nil, false
}

func (g *GraphVizGraph) Close() error {
	gErr := g.g.Close()
	gcErr := g.cg.Close()

	// TODO: move to multi-error.
	var err error
	if gErr != nil {
		err = fmt.Errorf("failed to close graphviz: %w", gErr)
	}

	if gcErr != nil {
		werr := fmt.Errorf("failed to close cgraph: %w", gcErr)

		if err != nil {
			return fmt.Errorf("%v: %w", werr, err)
		}

		return werr
	}

	return err
}

type GraphVizConfig struct {
	GraphVizOpts         []graphviz.GraphOption
	NodeColor, EdgeColor string
}

func buildLabelFromNode(n *Node) string {
	if n == nil {
		return ""
	}

	if n.IsOperandNode {
		return n.Operand
	}

	d, _ := n.Data.Float64()
	g, _ := n.Grad.Float64()

	return fmt.Sprintf("| data=%.4f | grad=%.4f |", d, g)
}

func buildLabelFromEdge(e *Edge) string {
	if e == nil {
		return ""
	}

	w, _ := e.Weight.Float64()
	b, _ := e.Bias.Float64()

	if w == 0 && b == 0 {
		return ""
	}

	return fmt.Sprintf("{ w=%.4f | b=%.4f }", w, b)
}

func stringOrDefault(s, d string) string {
	if s != "" {
		return s
	}

	return d
}
