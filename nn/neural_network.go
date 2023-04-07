package nn

func NewNeuralNetwork(cfg NeuralNetworkConfig) *NeuralNetwork {
	mlp := NewMLP(cfg.InputShape, cfg.Shape)

	return &NeuralNetwork{
		cfg: cfg,
		mlp: mlp,
	}
}

type NeuralNetworkConfig struct {
	InputShape  int
	Shape       []int
	WithGrapher bool
}

type Optimizer interface {
	Optimize([]*Value)
}

type NeuralNetwork struct {
	Optimizer Optimizer
	cfg       NeuralNetworkConfig
	mlp       *MLP
}

func (n *NeuralNetwork) Forward(inputs []*Value) {}

func (n *NeuralNetwork) Backprop() {}

func (n *NeuralNetwork) Optimize() {
	params := n.mlp.Parameters()
	n.Optimizer.Optimize(params)
}

func (n *NeuralNetwork) Train() {}
