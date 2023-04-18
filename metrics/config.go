package metrics

type Config struct {
	Namespace string
	Subsystem string
	Labels    map[string]string

	LearningRateGaugeLabels         []string
	ModelEpochCounterLabels         []string
	ModelLossValueGaugeLabels       []string
	ModelStepLatencyHistogramLabels []string
	ModelPhaseGaugeLabels           []string
}

