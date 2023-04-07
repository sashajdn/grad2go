package graph

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func NewGraphServerHandler(g Grapher) *GraphServerHandler {
	gsh := &GraphServerHandler{
		grapher: g,
	}

	r := gin.Default()

	r.LoadHTMLGlob("templates/*.html")

	gsh.r = gsh.initRouter(r)
	return gsh
}

type GraphServerHandler struct {
	r       *gin.Engine
	once    sync.Once
	grapher Grapher
	logger  *zap.SugaredLogger
}

func (g *GraphServerHandler) Handler() http.Handler { return g.r }

func (g *GraphServerHandler) handleGETRender(c *gin.Context) {
	if g.grapher == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "graph not initialised"})
		return
	}

	buf, err := g.grapher.Render()
	if err != nil {
		g.logger.With(zap.Error(err)).Error("Failed to render graph")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to render graph"})
		return
	}

	c.HTML(http.StatusOK, "graph.html", gin.H{
		"image": buf,
	})
}

func (g *GraphServerHandler) initRouter(r *gin.Engine) *gin.Engine {
	g.once.Do(func() {
		rg := r.Group("/graph")

		// GET.
		rg.GET("/render", g.handleGETRender)
	})

	return r
}
