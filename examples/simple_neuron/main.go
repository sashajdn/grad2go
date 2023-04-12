package main

import (
	"context"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"grad2go/graph"
	"grad2go/nn"
	"grad2go/serverpool"

	"github.com/goccy/go-graphviz"
	"github.com/shopspring/decimal"
	"go.uber.org/zap"
)

func main() {
	gcfg := graph.GraphVizConfig{
		GraphVizOpts: []graphviz.GraphOption{graphviz.Directed},
	}

	logger, _ := zap.NewProduction()
	sugaredLogger := logger.Sugar()

	g, err := graph.NewGraphVizGraph(gcfg)
	if err != nil {
		sugaredLogger.With(zap.Error(err)).Fatal("Failed to create new graphviz graph")
	}

	gsh, err := graph.NewGraphServerHandler(g, sugaredLogger)
	if err != nil {
		sugaredLogger.With(zap.Error(err)).Error("Failed to create graph server")
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

	// Imitate running step through a neuron.
	go runStep(ctx, g, sugaredLogger, 10*time.Second)

	<-ctx.Done()

	sugaredLogger.Info("Starting graceful shutdown of serverpool...")
	serverPool.Shutdown()
	sugaredLogger.Info("Graceful shutdown of serverpool complete")
}

func neuron(x, w, b float64) func() *nn.Value {
	return func() *nn.Value {
		xx := nn.NewValueWithLabel(decimal.NewFromFloat(x), nn.OperationNOOP, "x")
		ww := nn.NewValueWithLabel(decimal.NewFromFloat(w), nn.OperationNOOP, "w")
		bb := nn.NewValueWithLabel(decimal.NewFromFloat(b), nn.OperationNOOP, "b")

		out := xx.Mul(ww).Add(bb)

		// ReLu( w * x + b)
		activation := out.ReLu()

		return activation
	}
}

func runStep(ctx context.Context, g graph.Grapher, logger *zap.SugaredLogger, rate time.Duration) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set a do while rate so we can force one run straight away.
	doWhileRate := 1 * time.Microsecond

	for i := 0; ; i++ {
		select {
		case <-time.After(doWhileRate):
			x := r.Float64()
			w := r.Float64()
			b := r.Float64()

			logger.With(zap.Int("step_count", i)).Info("Running step...")

			forward := neuron(x, w, b)
			activation := forward()
			activation.Backward()

			if err := nn.BuildGraphFromRootValue(g, activation); err != nil {
				logger.With(zap.Error(err)).Fatal("Faield to build graph")

			}

			doWhileRate = rate
		case <-ctx.Done():
			return
		}
	}
}
