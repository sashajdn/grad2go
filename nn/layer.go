package nn

import "strconv"

func NewLayer(numberOfInputs, numberOfOutputs int) *Layer {
	var neurons = make([]*Neuron, 0, numberOfInputs)
	for i := 0; i < numberOfInputs; i++ {
		neurons = append(neurons, NewNeuron(numberOfOutputs))
	}

	return &Layer{
		neurons: neurons,
	}
}

// TODO: pass context & not label.
func NewLayerWithLabel(numberOfInputs, numberOfOutputs int, id int) *Layer {
	var neurons = make([]*Neuron, 0, numberOfInputs)
	for i := 0; i < numberOfInputs; i++ {
		context := &context{
			Layer:  id,
			Neuron: strconv.Itoa(i),
		}
		neurons = append(neurons, NewNeuronWithContext(numberOfOutputs, context))
	}

	return &Layer{
		neurons: neurons,
		id:      id,
	}
}

type Layer struct {
	neurons []*Neuron
	id      int
}

func (l *Layer) Forward(inputs []*Value) []*Value {
	var out = make([]*Value, 0, len(l.neurons))
	for _, n := range l.neurons {
		out = append(out, n.Forward(inputs))
	}

	return out
}

func (l *Layer) Parameters() []*Value {
	var out = make([]*Value, 0)

	for _, n := range l.neurons {
		out = append(out, n.Parameters()...)
	}

	return out
}
