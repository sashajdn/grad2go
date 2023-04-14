package nn

import (
	"errors"
	"fmt"
	"sync"
)

var (
	ErrInvalidNeuralNetworkPhase = errors.New("invalid neural network phase")
)

type Phase int8

const (
	PhaseStatic Phase = iota + 1
	PhaseForward
	PhaseBackward
	PhaseOptimize
)

func (n Phase) String() string {
	switch n {
	case PhaseStatic:
		return "static"
	case PhaseForward:
		return "forward"
	case PhaseBackward:
		return "backward"
	case PhaseOptimize:
		return "optimize"
	default:
		return "unknown"
	}
}

func NewNeuralNetwork(cfg NeuralNetworkConfig, optimizer Optimizer, losser Losser) *NeuralNetwork {
	mlp := NewMLP(cfg.InputShape, cfg.Shape)

	return &NeuralNetwork{
		Optimizer: optimizer,
		Losser:    losser,
		cfg:       cfg,
		mlp:       mlp,
		phase:     PhaseStatic,
	}
}

type NeuralNetworkConfig struct {
	InputShape int
	Shape      []int
}

type Optimizer func(input []*Value)

type Losser func(output, expectation []*Value) (*Value, error)

type NeuralNetwork struct {
	Optimizer     Optimizer
	Losser        Losser
	cfg           NeuralNetworkConfig
	mlp           *MLP
	phase         Phase
	phaseMu       sync.RWMutex
	outputStore   []*Value
	outputStoreMu sync.RWMutex
}

func (n *NeuralNetwork) Step(input, expectation []*Value) (*Value, error) {
	if err := n.forward(input); err != nil {
		return nil, fmt.Errorf("forward step failed: %w", err)
	}

	n.outputStoreMu.RLock()
	output := n.outputStore
	n.outputStoreMu.RUnlock()

	// TODO: we can check shape beforehand as this is the likely cause of error.
	loss, err := n.Losser(output, expectation)
	if err != nil {
		return nil, fmt.Errorf("failed to perform loss function: %w", err)
	}

	if err := n.backpropagation(loss); err != nil {
		return nil, fmt.Errorf("backpropagation step failed: %w", err)
	}

	if err := n.optimize(); err != nil {
		return nil, fmt.Errorf("optimize step failed: %w", err)
	}

	n.setPhase(PhaseStatic)

	return loss, nil
}

func (n *NeuralNetwork) Phase() Phase {
	n.phaseMu.RLock()
	defer n.phaseMu.RUnlock()

	return n.phase
}

func (n *NeuralNetwork) InputShape() int {
	return n.cfg.InputShape
}

func (n *NeuralNetwork) OutputShape() int {
	if len(n.cfg.Shape) == 0 {
		return 0
	}

	return n.cfg.Shape[len(n.cfg.Shape)-1]
}

func (n *NeuralNetwork) Shape() []int {
	return n.cfg.Shape
}

func (n *NeuralNetwork) Layers() int {
	return len(n.cfg.Shape)
}

func (n *NeuralNetwork) HiddenLayers() int { return n.Layers() - 1 }

func (n *NeuralNetwork) setPhase(newPhase Phase) {
	n.phaseMu.Lock()
	defer n.phaseMu.Unlock()

	n.phase = newPhase
}

func (n *NeuralNetwork) forward(inputs []*Value) error {
	// TODO: move to state machine.
	if n.phase != PhaseStatic {
		return fmt.Errorf(
			"cannot do forward pass, invalid phase %s must be static: %w",
			n.phase,
			ErrInvalidNeuralNetworkPhase,
		)
	}
	n.setPhase(PhaseForward)

	n.outputStoreMu.Lock()
	defer n.outputStoreMu.Unlock()
	n.outputStore = n.mlp.Forward(inputs)

	return nil
}

func (n *NeuralNetwork) backpropagation(lossValue *Value) error {
	if n.phase != PhaseForward {
		return fmt.Errorf(
			"cannot do backward pass, invalid phase %s must be forward: %w",
			n.phase,
			ErrInvalidNeuralNetworkPhase,
		)
	}
	n.setPhase(PhaseBackward)

	lossValue.Backward()

	return nil
}

func (n *NeuralNetwork) optimize() error {
	if n.phase != PhaseBackward {
		return fmt.Errorf(
			"cannot do optimize pass, invalid phase %s must be backward: %w",
			n.phase,
			ErrInvalidNeuralNetworkPhase,
		)
	}
	n.setPhase(PhaseOptimize)

	params := n.mlp.Parameters()
	n.Optimizer(params)

	return nil
}
