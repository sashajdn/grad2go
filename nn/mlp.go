package nn

import "strconv"

func NewMLP(numberOfInputs int, outputSizes []int) *MLP {
	var sizes = make([]int, 0, 1+len(outputSizes))
	sizes = append(sizes, numberOfInputs)
	sizes = append(sizes, outputSizes...)

	var layers = make([]*Layer, 0, len(outputSizes))
	for i := 0; i < len(outputSizes); i++ {
		layers = append(layers, NewLayerWithLabel(sizes[i], sizes[i+1], strconv.Itoa(i)))
	}

	return &MLP{
		layers: layers,
	}
}

type MLP struct {
	layers []*Layer
}

func (m *MLP) Forward(inputs []*Value) []*Value {
	var out = inputs
	for _, l := range m.layers {
		out = l.Forward(out)
	}

	return out
}

func (m *MLP) Parameters() []*Value {
	var out = make([]*Value, 0, len(m.layers))
	for _, l := range m.layers {
		out = append(out, l.Parameters()...)
	}

	return out
}
