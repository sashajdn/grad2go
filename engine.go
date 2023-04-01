package grad2go

import (
	"fmt"
	"log"

	"github.com/shopspring/decimal"
)

type Operation int32

const (
	OperationNOOP Operation = iota
	OperationAdd
	OperationSub
	OperationMul
	OperationDiv
	OperationExp
	OperationPow
)

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
	case OperationExp:
		return "exp"
	case OperationPow:
		return "pow"
	default:
		return "unknown"
	}
}

var noop = func() {}

func NewValue(value decimal.Decimal, operation Operation, children ...*Value) *Value {
	return &Value{
		value:     value,
		operation: operation,
		previous:  children,
		backward:  noop,
		grad:      decimal.NewFromFloat(0),
	}
}

type Value struct {
	value     decimal.Decimal
	operation Operation
	previous  []*Value
	backward  func()
	grad      decimal.Decimal
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
	case OperationExp:
		op = "exp"
	case OperationPow:
		op = "**"
	}

	va, _ := v.value.Float64()
	gr, _ := v.grad.Float64()

	return fmt.Sprintf("Value: %.3f, Op: %s, Grad: %.3f", va, op, gr)
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.value.Add(other.value), OperationAdd, v, other)

	v.backward = func() {
		v.grad = v.grad.Add(other.grad)
		out.grad = out.grad.Add(other.grad)
	}

	return out
}

func (v *Value) Sub(other *Value) *Value {
	unary := decimal.NewFromFloat(-1.0)
	unaryOther := NewValue(other.value.Mul(unary), OperationSub, other.previous...)
	return v.Add(unaryOther)
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.value.Mul(other.value), OperationMul, v, other)

	v.backward = func() {
		// Chain Rule: gradient at out node * differential over (v * other) w.r.t v.
		dvdout := other.value.Mul(out.grad)
		v.grad = v.grad.Add(dvdout)

		// Chain Rule: gradient at out node * differential over (v * other) w.r.t other.
		dodout := v.value.Mul(out.grad)
		other.grad = other.grad.Add(dodout)
	}

	return out
}

func (v *Value) Div(other *Value) *Value {
	if v, _ := other.value.Float64(); v == 0 {
		log.Fatalf("Division by zero error; other is zero")
	}

	inverse := decimal.NewFromFloat(1.0)
	inverseOther := NewValue(inverse.Div(other.value), OperationDiv, other.previous...)
	return v.Mul(inverseOther)
}

func (v *Value) Exp(other *Value) *Value {
	return nil
}

func (v *Value) Pow(other *Value) *Value {
}
