package serverpool

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

func NewHTTPServerPoolItem(name string, httpServer *http.Server) *HTTPServerPoolItem {
	return &HTTPServerPoolItem{
		s: httpServer,
        name: name,
	}
}

type HTTPServerPoolItem struct {
	s *http.Server
    name string
}

func (h *HTTPServerPoolItem) Start(ctx context.Context) error {
	err := h.s.ListenAndServe()
	switch {
	case errors.Is(err, http.ErrServerClosed):
		// TODO: log
	case err != nil:
		return fmt.Errorf("failed to listen & serve: %w", err)
	}

	return nil
}

func (h *HTTPServerPoolItem) Shutdown(ctx context.Context) error {
	return h.s.Shutdown(ctx)
}

func (h *HTTPServerPoolItem) Name() string { return h.name }
