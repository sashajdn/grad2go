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

func NewGraphVizGrapher(cfg GraphVizConfig) (*GraphVizGrapher, error) {
	gv := graphviz.New()
	g, err := gv.Graph(cfg.GraphVizOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create new graph: %w", err)
	}

	return &GraphVizGrapher{
		cnodes: make(map[string]*cgraph.Node),
		nodes:  make([]*Node, 0),
		edges:  make([]*Edge, 0),
		g:      graphviz.New(),
		cg:     g,
	}, nil
}

var _ Grapher = new(GraphVizGrapher)

// GraphVizGrapher ...
type GraphVizGrapher struct {
	cfg     GraphVizConfig
	cnodes  map[string]*cgraph.Node
	nodes   []*Node
	nodesMu sync.RWMutex
	edges   []*Edge
	cedges  map[string]*cgraph.Edge
	g       *graphviz.Graphviz
	cg      *cgraph.Graph
}

func (g *GraphVizGrapher) Render() error {
	var buf bytes.Buffer
	if err := g.g.Render(g.cg, graphviz.SVG, &buf); err != nil {
		return fmt.Errorf("failed to render graph to buffer: %w", err)
	}

	return nil
}

func (g *GraphVizGrapher) AddNode(n *Node) error {
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

func (g *GraphVizGrapher) AddEdge(n, m *Node, e *Edge) error {
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

	label := buildLabelFromNode(n)
	cedge.SetLabel(label)
	cedge.SetColor(stringOrDefault(g.cfg.EdgeColor, defaultEdgeColor))

	g.cedges[e.ID] = cedge // TODO: take lock
	g.edges = append(g.edges, e)

	return nil
}

func (g *GraphVizGrapher) setNode(id string, n *cgraph.Node) {
	g.nodesMu.Lock()
	defer g.nodesMu.Unlock()

	_, ok := g.cnodes[id]
	if ok {
		return
	}

	g.cnodes[id] = n
}

func (g *GraphVizGrapher) readNode(id string) (*cgraph.Node, bool) {
	g.nodesMu.RLock()
	defer g.nodesMu.RUnlock()

	if n, ok := g.cnodes[id]; ok {
		return n, true
	}

	return nil, false
}

func (g *GraphVizGrapher) Close() error {
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

	d, _ := n.Data.Float64()
	g, _ := n.Grad.Float64()

	return fmt.Sprintf("| data=%.4f | grad=%.4f | op=%s |", d, g, n.Operand)
}

func buildLabelFromEdge(e *Edge) string {
	if e == nil {
		return ""
	}

	w, _ := e.Weight.Float64()
	b, _ := e.Bias.Float64()

	return fmt.Sprintf("| w=%.4f | b=%.4f |", w, b)
}

func stringOrDefault(s, d string) string {
	if s != "" {
		return s
	}

	return d
}
