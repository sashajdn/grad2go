package grad2go

import "github.com/shopspring/decimal"

func max(a, b decimal.Decimal) decimal.Decimal {
	if a.GreaterThanOrEqual(b) {
		return a
	}

	return b
}
