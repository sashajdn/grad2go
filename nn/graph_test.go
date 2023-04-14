package nn

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBuildGraphVizLayersString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		layers         int
		expectedOutput string
	}{
		{
			name:           "1",
			layers:         1,
			expectedOutput: "0",
		},
		{
			name:           "2",
			layers:         2,
			expectedOutput: "0:1",
		},
		{
			name:           "5",
			layers:         5,
			expectedOutput: "0:1:2:3:4",
		},
	}

	for _, tt := range tests {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			res := BuildGraphVizLayersString(tt.layers)

			assert.Equal(t, tt.expectedOutput, res, "got %s, expected %s", res, tt.expectedOutput)
		})
	}
}
