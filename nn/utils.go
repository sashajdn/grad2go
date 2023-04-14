package nn

import (
	"github.com/shopspring/decimal"
)

func max(a, b decimal.Decimal) decimal.Decimal {
	if a.GreaterThanOrEqual(b) {
		return a
	}

	return b
}

func zip[T any](a, b []T, defaultValue T) [][]T {
	var out = make([][]T, 0, maxInt(len(a), len(b)))

	for i := 0; i < maxInt(len(a), len(b)); i++ {
		t := make([]T, 2)

		t[0] = defaultValue
		if i < len(a) {
			t[0] = a[i]
		}

		t[1] = defaultValue
		if i < len(b) {
			t[1] = b[i]
		}
	}

	return out

}

func maxInt(a, b int) int {
	if a > b {
		return a
	}

	return b
}
