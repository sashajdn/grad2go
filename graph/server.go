package graph

import (
	"embed"
	"fmt"
	"html/template"
	"net/http"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

const (
	templatesDir = "templates"
)

//go:embed templates/*
var fs embed.FS

func NewGraphServerHandler(g Grapher, logger *zap.SugaredLogger) (*GraphServerHandler, error) {
	logger = logger.With("object", "graph_server_handler")

	gsh := &GraphServerHandler{
		grapher: g,
		logger:  logger,
	}

	r := gin.Default()

	dfs, err := fs.ReadDir(templatesDir)
	if err != nil {
		return nil, fmt.Errorf("failed to load graph templates: %w", err)
	}

	var templates []string
	for _, df := range dfs {
		if df.IsDir() {
			continue
		}

		logger.With(zap.String("template", df.Name())).Info("Adding HTML template")
		templates = append(templates, df.Name())
	}

	r.LoadHTMLFiles(templates...)

	gsh.r = gsh.initRouter(r)
	return gsh, nil
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
		"image": template.HTML(cleanSVGString(buf.String())),
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

func cleanSVGString(s string) string {
	s = strings.TrimPrefix(s, `"`)
	s = strings.TrimSuffix(s, `"`)
	s = strings.TrimPrefix(s, `<?xml version="1.0" encoding="UTF-8" standalone="no"?>`)
	return s
}
