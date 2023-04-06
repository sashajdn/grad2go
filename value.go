package grad2go

import (
	"fmt"
	"log"
	"time"

	"github.com/shopspring/decimal"
)

var (
	// Since these values are used often; we might as well initialize rather than
	// having to create every function call.
	zero = decimal.NewFromFloat(0.0)
	one  = decimal.NewFromFloat(1.0)
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
		return "add"
	case OperationSub:
		return "sub"
	case OperationMul:
		return "mul"
	case OperationDiv:
		return "div"
	case OperationPow:
		return "pow"
	case OperationReLu:
		return "relu"
	default:
		return "unknown"
	}
}

var noop = func() {}

func NewValue(value decimal.Decimal, operation Operation, children ...*Value) *Value {
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
		operation: operation,
		previous:  previousSet,
		backward:  noop,
		grad:      decimal.NewFromFloat(0.0),
		id:        time.Now().UnixNano(),
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
	data      decimal.Decimal
	operation Operation
	previous  []*Value
	backward  func()
	grad      decimal.Decimal
	id        int64
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
	out := NewValue(v.data.Add(other.data), OperationAdd, v, other)

	out.backward = func() {
		v.grad = v.grad.Add(out.grad)
		other.grad = other.grad.Add(out.grad)
	}

	return out
}

func (v *Value) Sub(other *Value) *Value {
	out := NewValue(v.data.Sub(other.data), OperationSub, v, other)

	out.backward = func() {
		v.grad = v.grad.Add(out.grad)
		other.grad = other.grad.Add(out.grad)
	}

	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data.Mul(other.data), OperationMul, v, other)

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
	inverseOther := NewValue(inverse.Div(other.data), OperationDiv, other.previous...)

	out := v.Mul(inverseOther)
	out.operation = OperationDiv
	return out
}

func (v *Value) Pow(x decimal.Decimal) *Value {
	out := NewValue(v.data.Pow(x), OperationPow, v)

	out.backward = func() {
		dvdout := x.Mul(v.data.Pow(x.Sub(one)))
		v.grad = v.grad.Add(dvdout.Mul(out.grad))
	}

	return out
}

func (v *Value) ReLu() *Value {
	out := NewValue(max(zero, v.data), OperationPow, v)

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
