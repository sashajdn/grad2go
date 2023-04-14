package loss

import (
	"errors"
	"fmt"
	"grad2go/nn"

	"github.com/shopspring/decimal"
)

var (
	ErrInvalidShape = errors.New("invalid shape")
	ErrEmptyOutput  = errors.New("empty output")
)

func MeanSquaredError(output, expectation []*nn.Value) (*nn.Value, error) {
	if len(output) == 0 {
		return nil, ErrEmptyOutput
	}

	if len(output) != len(expectation) {
		return nil, fmt.Errorf("expected shape %d, got %d: %w", len(output), len(expectation), ErrInvalidShape)
	}

	// 1 / N * sum((y - yHat) ** 2).
	var intermediary = make([]*nn.Value, len(output))
	for i := 0; i < len(output); i++ {
		y, yHat := output[i], expectation[i]

		diff := y.Sub(yHat)
		squaredDiff := diff.Pow(decimal.NewFromInt(2))

		intermediary[i] = squaredDiff
	}

	summation := intermediary[0]
	for i := 1; i < len(intermediary); i++ {
		next := intermediary[i]
		summation = summation.Add(next)
	}

	divisor := nn.NewValueWithLabel(decimal.NewFromInt(
		int64(len(output))),
		nn.OperationNOOP,
		"mse_divisor",
	)

	out := summation.Div(divisor)
	return out, nil
}
