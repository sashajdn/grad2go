package grad2go

import (
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
	}{
		{
			name:          "simple_int_add",
			a:             NewValue(decimal.NewFromFloat(1.0), OperationNOOP),
			b:             NewValue(decimal.NewFromFloat(1.0), OperationNOOP),
			expectedValue: NewValue(decimal.NewFromFloat(1.0), OperationAdd),
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
		},
	}

	for _, tt := range tests {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
		})

		c := tt.op(tt.a, tt.b)
		c.Backward()

		expectedValue := tt.applier(tt.expectedValue, tt.operand, tt.a, tt.b)

		assert.Equal(t, expectedValue.data, c.data, "expected: %v, got: %v", expectedValue.data, c.data)
		assert.Equal(t, expectedValue.grad, c.grad, "expected: %v, got: %v", expectedValue.grad, c.grad)
		assert.Equal(t, expectedValue.previous, c.previous, "expected: %v, got: %v", expectedValue.previous, c.previous)
		assert.Equal(t, expectedValue.operation, c.operation, "expected: %v, got: %v", expectedValue.operation, c.operation)
	}
}
