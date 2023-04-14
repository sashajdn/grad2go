package nn

import (
	"log"
	"math/rand"
	"time"

	"github.com/shopspring/decimal"
)

var minusOne = decimal.NewFromFloat(-1.0)

func NewNeuron(numberOfInputs int) *Neuron {
	n := &Neuron{
		r: rand.New(rand.NewSource(time.Now().UnixNano())),
		d: numberOfInputs,
	}

	n.W = n.generateRandomVector(numberOfInputs, 1, KindWeight)
	n.B = n.generateRandomVector(1, 1, KindBias)

	return n
}

func NewNeuronWithContext(numberOfInputs int, context *context) *Neuron {
	// TODO: consolidate with the above.
	n := &Neuron{
		r:       rand.New(rand.NewSource(time.Now().UnixNano())),
		d:       numberOfInputs,
		context: context,
	}

	n.W = n.generateRandomVector(numberOfInputs, 1, KindWeight)
	n.B = n.generateRandomVector(1, 1, KindBias)

	return n
}

type Neuron struct {
	W       []*Value
	B       []*Value
	r       *rand.Rand
	d       int
	context *context
}

func (n *Neuron) Forward(inputs []*Value) *Value {
	if len(inputs) != n.d {
		log.Fatalf("invalid dim of inputs: got %d, expected %d", len(inputs), n.d)
	}

	// w * x + b
	var sum = n.B[0]
	for i := 0; i < n.d; i++ {
		w := n.W[i]
		x := inputs[i]
		product := w.Mul(x)

		sum = sum.Add(product)
	}
	activation := sum.ReLu()

	return activation
}

func (n *Neuron) Parameters() []*Value {
	// TODO: copy values.
	var out = make([]*Value, 0, len(n.W)+len(n.B))
	out = append(out, n.W...)
	out = append(out, n.B...)

	return out
}

func (n *Neuron) generateRandomVector(size int, linSpace float64, kind Kind) []*Value {
	switch {
	case linSpace == 0:
		linSpace = 1
	case linSpace < 0:
		linSpace = -1.0 * linSpace
	}

	var out = make([]*Value, size)

	for i := 0; i < size; i++ {
		negative := n.r.Float64() > 0.5
		value := decimal.NewFromFloat(n.r.Float64())

		if linSpace != 1 {
			rangeMul := decimal.NewFromFloat(linSpace)
			value = value.Mul(rangeMul)
		}

		if negative {
			value = value.Mul(minusOne)
		}

		out[i] = newValueWithContext(value, OperationNOOP, kind, n.context)
	}

	return out
}
