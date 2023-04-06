package grad2go

func NewNeuralNetwork(cfg NeuralNetworkConfig) *NeuralNetwork {
	mlp := NewMLP(cfg.InputShape, cfg.Shape)

	return &NeuralNetwork{
		cfg: cfg,
		mlp: mlp,
	}
}

type NeuralNetworkConfig struct {
	InputShape int
	Shape      []int
}

type NeuralNetwork struct {
	cfg NeuralNetworkConfig
	mlp *MLP
}

func (n *NeuralNetwork) Forward(inputs []*Value) {}

func (n *NeuralNetwork) Backprop() {}

func (n *NeuralNetwork) SGD() {}

func (n *NeuralNetwork) Train() {}
