package nn

import (
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/shopspring/decimal"
)

var (
	// Since these values are used often; we might as well initialize rather than
	// having to create every function call.
	zero = decimal.NewFromFloat(0.0)
	one  = decimal.NewFromFloat(1.0)
)

type Kind int32

const (
	KindUnknown Kind = iota
	KindBias
	KindWeight
	KindInput
	KindValue
)

type Operation int32

const (
	OperationNOOP Operation = iota
	OperationAdd
	OperationSub
	OperationMul
	OperationDiv
	OperationPow
	OperationReLu
)

// String implements the stringer interface.
func (o Operation) String() string {
	switch o {
	case OperationNOOP:
		return "noop"
	case OperationAdd:
		return "+"
	case OperationSub:
		return "-"
	case OperationMul:
		return "*"
	case OperationDiv:
		return "/"
	case OperationPow:
		return "**"
	case OperationReLu:
		return "ReLu"
	default:
		return "unknown"
	}
}

var noop = func() {}

func NewValue(
	value decimal.Decimal,
	operation Operation,
	kind Kind,
	label string,
	children ...*Value,
) *Value {
	context := &context{
		Label: label,
	}

	return newValueWithContext(value, operation, kind, context, children...)
}

func newValueWithContext(
	value decimal.Decimal,
	operation Operation,
	kind Kind,
	context *context,
	children ...*Value,
) *Value {
	var (
		previousSet []*Value
		previousMap = make(map[*Value]struct{}, len(children))
	)

	for _, child := range children {
		if _, ok := previousMap[child]; ok {
			continue
		}

		previousMap[child] = struct{}{}
		previousSet = append(previousSet, child)
	}

	return &Value{
		data:      value,
		kind:      kind,
		operation: operation,
		previous:  previousSet,
		backward:  noop,
		grad:      decimal.NewFromFloat(0.0),
		id:        time.Now().UnixNano(),
		context:   context,
	}
}

func (v *Value) Backward() {
	var (
		s    = map[*Value]struct{}{}
		topo []*Value
	)

	var collect func(node *Value)
	collect = func(node *Value) {
		if _, ok := s[node]; ok {
			return
		}

		s[node] = struct{}{}
		for _, c := range node.previous {
			collect(c)
		}

		topo = append(topo, node)
	}
	collect(v)

	v.grad = one
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		node.backward()
	}
}

type Value struct {
	kind      Kind
	data      decimal.Decimal
	operation Operation
	previous  []*Value
	backward  func()
	grad      decimal.Decimal
	id        int64
	context   *context
}

func (v *Value) Label() string {
	if v.context == nil {
		return ""
	}

	return v.context.String()
}

func (v *Value) layer() int {
	if v.context == nil {
		return -1
	}

	return v.context.Layer
}

func (v *Value) String() string {
	var op = "noop"
	switch v.operation {
	case OperationAdd:
		op = "+"
	case OperationMul:
		op = "*"
	case OperationSub:
		op = "-"
	case OperationDiv:
		op = "/"
	case OperationPow:
		op = "**"
	case OperationReLu:
		op = "relu"
	}

	va, _ := v.data.Float64()
	gr, _ := v.grad.Float64()

	return fmt.Sprintf("Value: %.3f, Op: %s, Grad: %.3f", va, op, gr)
}

func (v *Value) Add(other *Value) *Value {
	mergedContext := mergeContexts(v.context, other.context)
	out := newValueWithContext(v.data.Add(other.data), OperationAdd, KindValue, mergedContext, v, other)

	out.backward = func() {
		v.grad = v.grad.Add(out.grad)
		other.grad = other.grad.Add(out.grad)
	}

	return out
}

func (v *Value) Sub(other *Value) *Value {
	mergedContext := mergeContexts(v.context, other.context)
	out := newValueWithContext(v.data.Sub(other.data), OperationSub, KindValue, mergedContext, v, other)

	out.backward = func() {
		v.grad = v.grad.Add(out.grad)
		other.grad = other.grad.Add(out.grad)
	}

	return out
}

func (v *Value) Mul(other *Value) *Value {
	mergedContext := mergeContexts(v.context, other.context)
	out := newValueWithContext(v.data.Mul(other.data), OperationMul, KindValue, mergedContext, v, other)

	out.backward = func() {
		// Chain Rule: gradient at out node * differential over (v * other) w.r.t v.
		dvdout := other.data.Mul(out.grad)
		v.grad = v.grad.Add(dvdout)

		// Chain Rule: gradient at out node * differential over (v * other) w.r.t other.
		dodout := v.data.Mul(out.grad)
		other.grad = other.grad.Add(dodout)
	}

	return out
}

func (v *Value) Div(other *Value) *Value {
	// TODO: validate.
	if v, _ := other.data.Float64(); v == 0 {
		log.Fatalf("Division by zero error; other is zero")
	}

	inverse := decimal.NewFromFloat(1.0)
	mergedContext := mergeContexts(v.context, other.context)
	inverseOther := newValueWithContext(inverse.Div(other.data), OperationDiv, KindValue, mergedContext, other.previous...)

	out := v.Mul(inverseOther)
	out.operation = OperationDiv
	return out
}

func (v *Value) Pow(x decimal.Decimal) *Value {
	out := newValueWithContext(v.data.Pow(x), OperationPow, KindValue, v.context, v)

	out.backward = func() {
		dvdout := x.Mul(v.data.Pow(x.Sub(one)))
		v.grad = v.grad.Add(dvdout.Mul(out.grad))
	}

	return out
}

func (v *Value) ReLu() *Value {
	out := newValueWithContext(max(zero, v.data), OperationReLu, KindValue, v.context, v)

	out.backward = func() {
		binary := func() decimal.Decimal {
			if out.data.GreaterThan(zero) {
				return one
			}

			return zero
		}()

		v.grad = v.grad.Add(binary.Mul(out.grad))
	}

	return out
}

func (v *Value) Float64() float64 {
	f, _ := v.data.Float64()
	return f
}

func (v *Value) ApplyDescent(rate decimal.Decimal) {
	apply := v.data.Mul(rate)
	v.data = v.data.Add(apply)
}

func (v *Value) ID() string { return strconv.Itoa(int(v.id)) }
func (v *Value) Kind() Kind { return v.kind }
