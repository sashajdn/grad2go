package grad2go

func NewLayer(numberOfInputs, numberOfOutputs int) *Layer {
	var neurons = make([]*Neuron, 0, numberOfInputs)
	for i := 0; i < numberOfInputs; i++ {
		neurons = append(neurons, NewNeuron(numberOfOutputs))
	}

	return &Layer{
		neurons: neurons,
	}
}

type Layer struct {
	neurons []*Neuron
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
