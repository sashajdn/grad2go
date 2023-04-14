package main

import (
	"context"
	"fmt"
	"grad2go/graph"
	"grad2go/nn"
	"grad2go/optimizer"
	"grad2go/serverpool"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
	"github.com/shopspring/decimal"
	"go.uber.org/zap"
)

func main() {
	gcfg := graph.GraphVizConfig{
		GraphVizOpts: []graphviz.GraphOption{graphviz.Directed},
		RankDir:      cgraph.LRRank,
	}

	logger, _ := zap.NewProduction()
	sugaredLogger := logger.Sugar()

	g, err := graph.NewGraphVizGraph(gcfg)
	if err != nil {
		sugaredLogger.With(zap.Error(err)).Fatal("Failed to create new graphviz graph")
	}

	gsh, err := graph.NewGraphServerHandler(g, sugaredLogger)
	if err != nil {
		sugaredLogger.With(zap.Error(err)).Fatal("Failed to create graph server")
	}

	httpServer := &http.Server{
		Addr:    "0.0.0.0:8080",
		Handler: gsh.Handler(),
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill, syscall.SIGTERM)
	defer cancel()

	serverPool := serverpool.New(sugaredLogger)
	serverPool.Add(serverpool.NewHTTPServerPoolItem("grapher", httpServer))

	errCh := serverPool.Start(ctx)

	net := nn.NewNeuralNetwork(nn.NeuralNetworkConfig{
		InputShape: 3,
		Shape:      []int{3, 3},
	}, optimizer.SGD)

	// Handle errors.
	go func() {
		select {
		case err, ok := <-errCh:
			if !ok {
				return
			}

			sugaredLogger.With(zap.Error(err)).Error("Error received from server pool")
		case <-ctx.Done():
			return
		}

	}()

	// Run a step on the neural network.
	go runStep(ctx, net, g, sugaredLogger, 10*time.Second)

	<-ctx.Done()

	sugaredLogger.Info("Starting graceful shutdown of serverpool...")
	serverPool.Shutdown()
	sugaredLogger.Info("Graceful shutdown of serverpool complete")
}

func generateRandInput(r *rand.Rand, size int) []*nn.Value {
	var vv = make([]*nn.Value, size)
	for i := 0; i < size; i++ {

		// Generate random value from cont. set [-1, 1].
		f := r.Float64()
		b := func() float64 {
			if r.Float64() > 0.5 {
				return 1
			}

			return -1
		}()

		v := nn.NewValueWithLabel(decimal.NewFromFloat(f*b), nn.OperationNOOP, fmt.Sprintf("input_%d", i))
		vv[i] = v
	}

	return vv
}

func runStep(ctx context.Context, net *nn.NeuralNetwork, g graph.Grapher, logger *zap.SugaredLogger, rate time.Duration) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set a do while rate so we can force one run straight away.
	doWhileRate := 1 * time.Microsecond

	for i := 0; ; i++ {
		select {
		case <-time.After(doWhileRate):
		case <-ctx.Done():
			return
		}

		logger := logger.With(zap.Int("step_count", i))

		input := generateRandInput(r, net.InputShape())

		var logParams = make([]float64, 0, len(input))
		for _, in := range input {
			logParams = append(logParams, in.Float64())
		}

		logger.With(
			zap.Float64s("input", logParams),
		).Info("Running NN step")

		lossValue, err := net.Step(input)
		if err != nil {
			logger.With(zap.Error(err)).Error("Failed to perfom neural network step")
			continue
		}

		if err := nn.BuildGraphFromRootValue(g, lossValue); err != nil {
			logger.With(zap.Error(err)).Error("Failed to render graph for neural network step")
			continue
		}

		doWhileRate = rate
	}
}
