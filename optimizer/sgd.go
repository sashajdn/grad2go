package optimizer

import (
	"grad2go/nn"

	"github.com/shopspring/decimal"
)

const defaultLearningRate = 0.01

func SGD(values []*nn.Value) {
	for _, v := range values {
		v.ApplyDescent(decimal.NewFromFloat(-defaultLearningRate))
	}
}
