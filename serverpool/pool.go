package serverpool

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

const gracefulShutdownTimeout = 10 * time.Second

func New(logger *zap.SugaredLogger) *ServerPool {
	logger = logger.With(zap.String("object", "server_pool"))

	return &ServerPool{
		logger: logger,
	}
}

type ServerPool struct {
	pool   []ServerPoolItem
	poolMu sync.RWMutex
	logger *zap.SugaredLogger
	once   sync.Once
}

func (s *ServerPool) Add(server ServerPoolItem) {
	s.poolMu.Lock()
	defer s.poolMu.Lock()

	s.pool = append(s.pool, server)
	s.logger.With(zap.String("pool_item", server.Name())).Info("Adding server to server pool")
}

func (s *ServerPool) Start(ctx context.Context) (ch chan error) {
	s.once.Do(func() {
		ch = make(chan error, 1)

		for _, poolItem := range s.pool {
			// Move to context error group
			poolItem := poolItem
			go func() {
				id := poolItem.Name()

				s.logger.With(zap.String("pool_item", id)).Info("Starting server")
				if err := poolItem.Start(ctx); err != nil {
					select {
					case ch <- fmt.Errorf("failed to start server %s: %w", id, err):
					case <-time.After(5 * time.Second):
						s.logger.With(zap.Error(err)).Error("Failed to publish start server error")
					}
				}
			}()
		}
	})

	return ch
}

func (s *ServerPool) Shutdown() chan error {
	s.logger.Info("Shutdown called on server pool; graceful shutdown starting...")

	var wg sync.WaitGroup

	ctx, cancel := context.WithTimeout(context.Background(), gracefulShutdownTimeout)
	defer cancel()

	s.poolMu.RLock()
	ch := make(chan error, len(s.pool))

	for _, poolItem := range s.pool {
		wg.Add(1)
		poolItem := poolItem

		go func() {
			defer wg.Done()
			id := poolItem.Name()

			s.logger.With(zap.String("pool_item", id)).Warn("Gracefully shutting down pool item")
			if err := poolItem.Shutdown(ctx); err != nil {
				select {
				case ch <- fmt.Errorf("failed to shutdown server %s: %w", id, err):
				case <-time.After(10 * time.Second):
				}

				return
			}

			s.logger.With(zap.String("pool_item", id)).Warn("Pool item shutdown complete")
		}()
	}

	s.poolMu.Unlock()

	wg.Wait()
	return ch
}
