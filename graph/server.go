package graph

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

func NewGraphServerHandler(g Grapher) *GraphServerHandler {
	gsh := &GraphServerHandler{
		grapher: g,
	}
	gsh.r = gsh.router(gin.Default())
	return gsh
}

type GraphServerHandler struct {
	r       *gin.Engine
	once    sync.Once
	grapher Grapher
}

func (g *GraphServerHandler) Handler() http.Handler { return g.r }

func (g *GraphServerHandler) handleGETRender(c *gin.Context) {
	if g.grapher == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "graph not initialised"})
		return
	}

	// TODO: handle buf.
	_, err := g.grapher.Render()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to render graph"})
	}
}

func (g *GraphServerHandler) router(r *gin.Engine) *gin.Engine {
	g.once.Do(func() {
		rg := r.Group("/graph")

		// GET.
		rg.GET("/render", g.handleGETRender)
	})

	return r
}
