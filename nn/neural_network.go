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

func NewNeuralNetwork(cfg NeuralNetworkConfig, optimizer Optimizer) *NeuralNetwork {
	mlp := NewMLP(cfg.InputShape, cfg.Shape)

	return &NeuralNetwork{
		Optimizer: optimizer,
		cfg:       cfg,
		mlp:       mlp,
		phase:     PhaseStatic,
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
	Optimizer      Optimizer
	cfg            NeuralNetworkConfig
	mlp            *MLP
	phase          Phase
	phaseMu        sync.RWMutex
	forwardStore   []*Value
	forwardStoreMu sync.RWMutex
}

func (n *NeuralNetwork) Step(input []*Value) error {
	if err := n.forward(input); err != nil {
		return fmt.Errorf("forward step failed: %w", err)
	}

	if err := n.backpropagation(); err != nil {
		return fmt.Errorf("backpropagation step failed: %w", err)
	}

	if err := n.optimize(); err != nil {
		return fmt.Errorf("optimize step failed: %w", err)
	}

	n.setPhase(PhaseStatic)

	return nil
}

func (n *NeuralNetwork) Phase() Phase {
	n.phaseMu.RLock()
	defer n.phaseMu.RUnlock()

	return n.phase
}

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

	n.forwardStoreMu.Lock()
	defer n.forwardStoreMu.Unlock()
	n.forwardStore = n.mlp.Forward(inputs)

	return nil
}

func (n *NeuralNetwork) backpropagation() error {
	if n.phase != PhaseForward {
		return fmt.Errorf(
			"cannot do backward pass, invalid phase %s must be forward: %w",
			n.phase,
			ErrInvalidNeuralNetworkPhase,
		)
	}
	n.setPhase(PhaseBackward)

	for _, v := range n.forwardStore {
		v.Backward()
	}

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
	n.Optimizer.Optimize(params)

	return nil
}
