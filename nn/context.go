package nn

import "fmt"

type context struct {
	Label   string
	Neuron  string
	Layer   int
	Network string
}

func (c *context) String() string {
	if c == nil {
		return ""
	}

	return fmt.Sprintf(`
Label: %s
Neuron: %s
Layer: %d
Network: %s
`, c.Label, c.Neuron, c.Layer, c.Network)
}

func mergeContexts(a, b *context) *context {
	switch {
	case a == nil && b == nil:
		return nil
	case a == nil:
		return b
	case b == nil:
		return a
	}

	return &context{
		Label:   a.Label,
		Neuron:  a.Neuron,
		Layer:   maxInt(a.Layer, b.Layer),
		Network: a.Network,
	}
}
