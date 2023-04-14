package nn

import (
	"fmt"
	"testing"

	"github.com/shopspring/decimal"
	"github.com/stretchr/testify/assert"
)

func TestValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		a, b          *Value
		applier       func(c *Value, operand Operation, previous ...*Value) *Value
		op            func(a, b *Value) *Value
		operand       Operation
		expectedValue *Value
		expectedAGrad decimal.Decimal
		expectedBGrad decimal.Decimal
	}{
		{
			name: "simple_int_add",
			a:    newValueWithContext(decimal.NewFromFloat(1.0), OperationNOOP, KindValue, nil),
			b:    newValueWithContext(decimal.NewFromFloat(1.0), OperationNOOP, KindValue, nil),
			op: func(a, b *Value) *Value {
				return a.Add(b)
			},
			operand: OperationAdd,
			applier: func(c *Value, operand Operation, previous ...*Value) *Value {
				c.data = decimal.NewFromFloat(2.0)
				c.grad = decimal.NewFromFloat(1.0)
				c.previous = previous
				c.operation = operand
				return c
			},
			expectedValue: newValueWithContext(decimal.NewFromFloat(1.0), OperationAdd, KindValue, nil),
			expectedAGrad: decimal.NewFromFloat(1.0),
			expectedBGrad: decimal.NewFromFloat(1.0),
		},
		{
			name: "simple_int_mul",
			a:    newValueWithContext(decimal.NewFromFloat(2.0), OperationNOOP, KindValue, nil),
			b:    newValueWithContext(decimal.NewFromFloat(3.0), OperationNOOP, KindValue, nil),
			op: func(a, b *Value) *Value {
				return a.Mul(b)
			},
			operand: OperationMul,
			applier: func(c *Value, operand Operation, previous ...*Value) *Value {
				c.data = decimal.NewFromFloat(6.0)
				c.grad = decimal.NewFromFloat(1.0)
				c.previous = previous
				c.operation = operand
				return c
			},
			expectedValue: newValueWithContext(decimal.NewFromFloat(6.0), OperationMul, KindValue, nil),
			expectedAGrad: decimal.NewFromFloat(3.0),
			expectedBGrad: decimal.NewFromFloat(2.0),
		},
		{
			name: "simple_int_sub",
			a:    newValueWithContext(decimal.NewFromFloat(2.0), OperationNOOP, KindValue, nil),
			b:    newValueWithContext(decimal.NewFromFloat(3.0), OperationNOOP, KindValue, nil),
			op: func(a, b *Value) *Value {
				return a.Sub(b)
			},
			operand: OperationSub,
			applier: func(c *Value, operand Operation, previous ...*Value) *Value {
				c.data = decimal.NewFromFloat(-1.0)
				c.grad = decimal.NewFromFloat(1.0)
				c.previous = previous
				c.operation = operand
				return c
			},
			expectedValue: newValueWithContext(decimal.NewFromFloat(-1.0), OperationSub, KindValue, nil),
			expectedAGrad: decimal.NewFromFloat(1.0),
			expectedBGrad: decimal.NewFromFloat(1.0),
		},
	}

	for _, tt := range tests {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			c := tt.op(tt.a, tt.b)
			c.Backward()

			expectedValue := tt.applier(tt.expectedValue, tt.operand, tt.a, tt.b)

			fmt.Println(tt.a)
			fmt.Println(tt.b)
			fmt.Println(c)
			fmt.Println(expectedValue)

			assert.Equal(t, expectedValue.data, c.data, "data: expected: %v, got: %v", expectedValue.data, c.data)
			assert.Equal(t, expectedValue.grad, c.grad, "grad: expected: %v, got: %v", expectedValue.grad, c.grad)
			assert.Equal(t, expectedValue.previous, c.previous, "prev: expected: %v, got: %v", expectedValue.previous, c.previous)
			assert.Equal(t, expectedValue.operation, c.operation, "op: expected: %v, got: %v", expectedValue.operation, c.operation)

			assert.Equal(t, tt.expectedAGrad, tt.a.grad, "a grad: expected: %+v, got: %+v", tt.expectedAGrad, tt.a.grad)
			assert.Equal(t, tt.expectedBGrad, tt.b.grad, "b grad: expected: %+v, got: %+v", tt.expectedBGrad, tt.b.grad)
		})
	}
}
