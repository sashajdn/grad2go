package serverpool

import "context"

type ServerPoolItem interface {
	Start(ctx context.Context) error
	Shutdown(context.Context) error
	Name() string
}
