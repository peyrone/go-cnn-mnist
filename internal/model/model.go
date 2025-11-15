package model

import (
	"encoding/gob"
	"os"

	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

const (
	InputChannels = 1
	InputHeight   = 28
	InputWidth    = 28

	Conv1Filters = 8
	KernelSize   = 1

	// After Conv(1x1) + MaxPool(2x2, stride 2) on 28x28:
	// output = 8 x 14 x 14 -> flatten
	FlattenSize = Conv1Filters * 14 * 14 // = 8 * 14 * 14 = 1568

	HiddenSize = 64
	NumClasses = 10
)

// Holds all trainable parameters of the CNN
type ModelWeights struct {
	Wc1, Bc1   []float32
	Wfc1, Bfc1 []float32
	Wfc2, Bfc2 []float32
}

// Extract and copies []float32 from a Gorgonia node
func cloneData(n *G.Node) []float32 {
	raw := n.Value().Data().([]float32)
	out := make([]float32, len(raw))
	copy(out, raw)
	return out
}

// Pull current values from nodes and returns a ModelWeights
func CaptureWeights(
	wc1, bc1, wfc1, bfc1, wfc2, bfc2 *G.Node,
) *ModelWeights {
	return &ModelWeights{
		Wc1:  cloneData(wc1),
		Bc1:  cloneData(bc1),
		Wfc1: cloneData(wfc1),
		Bfc1: cloneData(bfc1),
		Wfc2: cloneData(wfc2),
		Bfc2: cloneData(bfc2),
	}
}

// Save writes the model to a file using gob
func Save(path string, m *ModelWeights) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(m)
}

// Load reads the model from a gob file
func Load(path string) (*ModelWeights, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var m ModelWeights
	if err := gob.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

// AssignWeights loads the weights into the graph nodes.
// Shapes MUST match your training architecture.
func AssignWeights(
	mw *ModelWeights,
	wc1, bc1, wfc1, bfc1, wfc2, bfc2 *G.Node,
) error {
	G.Let(wc1, T.New(
		T.WithBacking(mw.Wc1),
		T.WithShape(Conv1Filters, InputChannels, KernelSize, KernelSize),
	))
	G.Let(bc1, T.New(
		T.WithBacking(mw.Bc1),
		T.WithShape(Conv1Filters),
	))
	G.Let(wfc1, T.New(
		T.WithBacking(mw.Wfc1),
		T.WithShape(FlattenSize, HiddenSize),
	))
	G.Let(bfc1, T.New(
		T.WithBacking(mw.Bfc1),
		T.WithShape(HiddenSize),
	))
	G.Let(wfc2, T.New(
		T.WithBacking(mw.Wfc2),
		T.WithShape(HiddenSize, NumClasses),
	))
	G.Let(bfc2, T.New(
		T.WithBacking(mw.Bfc2),
		T.WithShape(NumClasses),
	))
	return nil
}
