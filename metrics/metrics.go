package metrics

import (
	"errors"
	"fmt"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	registry *prometheus.Registry
	once     sync.Once
	// TODO: move to atomic flag
	initialized bool
)

var (
	learningRateGauge *prometheus.GaugeVec
)
var (
	modelPhaseGauge           *prometheus.GaugeVec
	modelLossValueGauge       *prometheus.GaugeVec
	modelStepLatencyHistogram *prometheus.HistogramVec
	modelEpochCounter         *prometheus.CounterVec
)

var ErrMetricsNotInitialized = errors.New("metrics not initialized")

func Init(config Config) error {
	var err error
	once.Do(func() {
		initialized = true

		registry = prometheus.NewRegistry()

		learningRateGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace:   config.Namespace,
			Subsystem:   config.Subsystem,
			Name:        "learning_rate_gauge",
			Help:        "The gauge for the learning rate of the neural network",
			ConstLabels: config.Labels,
		}, config.LearningRateGaugeLabels)
		if err = registry.Register(learningRateGauge); err != nil {
			return
		}

		modelPhaseGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace:   config.Namespace,
			Subsystem:   config.Subsystem,
			Name:        "model_phase_gauge",
			Help:        "The gauge for the phase of which the model is currently in",
			ConstLabels: config.Labels,
		}, config.ModelPhaseGaugeLabels)
		if err = registry.Register(modelPhaseGauge); err != nil {
			return
		}

		modelLossValueGauge = prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace:   config.Namespace,
			Subsystem:   config.Subsystem,
			Name:        "model_loss_value_gauge",
			Help:        "The gauge for the model loss function value",
			ConstLabels: config.Labels,
		}, config.ModelLossValueGaugeLabels)
		if err = registry.Register(modelLossValueGauge); err != nil {
			return
		}

		modelStepLatencyHistogram = prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace:   config.Namespace,
			Subsystem:   config.Subsystem,
			Name:        "model_step_latency_histogram",
			Help:        "The histogram of latencies for the time taken to execute a full step of a model",
			ConstLabels: config.Labels,
		}, config.ModelStepLatencyHistogramLabels)
		if err = registry.Register(modelStepLatencyHistogram); err != nil {
			return
		}

		modelEpochCounter = prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace:   config.Namespace,
			Subsystem:   config.Subsystem,
			Name:        "model_epoch_counter",
			Help:        "The counter for the epoch of the model",
			ConstLabels: config.Labels,
		}, config.ModelEpochCounterLabels)
		if err = registry.Register(modelEpochCounter); err != nil {
			return
		}
	})

	if err != nil {
		return fmt.Errorf("failed to init prometheus client: %w", err)
	}

	return nil
}

func ObserveLearningRateGauge(learningRate float64, labels []string) error {
	if !initialized {
		return ErrMetricsNotInitialized
	}

	g, err := learningRateGauge.GetMetricWithLabelValues(labels...)
	if err != nil {
		return fmt.Errorf("failed to get learning rate gauge metric: %w", err)
	}

	g.Set(learningRate)
	return nil
}

func ObserveModelLossValueGauge(lossValue float64, labels []string) error {
	if !initialized {
		return ErrMetricsNotInitialized
	}

	g, err := modelLossValueGauge.GetMetricWithLabelValues(labels...)
	if err != nil {
		return fmt.Errorf("failed to get model loss gauge metric: %w", err)
	}

	g.Set(lossValue)
	return nil
}

func ObserveModelStepLatencyHistogram(latencyMS float64, labels []string) error {
	if !initialized {
		return ErrMetricsNotInitialized
	}

	h, err := modelStepLatencyHistogram.GetMetricWithLabelValues(labels...)
	if err != nil {
		return fmt.Errorf("failed to get model step latency histogram metric: %w", err)
	}

	h.Observe(latencyMS)
	return nil
}

func ObserveModelPhaseGauge(phase int, labels []string) error {
	if !initialized {
		return ErrMetricsNotInitialized
	}

	g, err := modelPhaseGauge.GetMetricWithLabelValues(labels...)
	if err != nil {
		return fmt.Errorf("failed to get model phase gauge metric: %w", err)
	}

	g.Set(float64(phase))
	return nil
}

func ObserveModelEpochCounter(labels []string) error {
	if !initialized {
		return ErrMetricsNotInitialized
	}

	g, err := modelEpochCounter.GetMetricWithLabelValues(labels...)
	if err != nil {
		return fmt.Errorf("failed to get model epoch counter metric: %w", err)
	}

	g.Inc()
	return nil
}
