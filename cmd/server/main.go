package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/peyrone/go-cnn-mnist/internal/model"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

const (
	modelPath = "best_model.gob"
)

type PredictRequest struct {
	// 28*28 float32 values in [0,1], row-major
	Pixels []float32 `json:"pixels"`
}

type PredictResponse struct {
	Digit int       `json:"digit"`
	Probs []float32 `json:"probs"`
}

func main() {
	// Load trained weights
	mw, err := model.Load(modelPath)
	if err != nil {
		log.Fatalf("cannot load model %q: %v", modelPath, err)
	}
	fmt.Printf("Loaded model from %s\n", modelPath)

	// Build inference graph: same architecture, batch size 1
	g := G.NewGraph()
	x := G.NewTensor(g, T.Float32, 4,
		G.WithName("x"),
		G.WithShape(1, model.InputChannels, model.InputHeight, model.InputWidth),
	)

	wc1 := G.NewTensor(g, T.Float32, 4,
		G.WithName("wc1"),
		G.WithShape(model.Conv1Filters, model.InputChannels, model.KernelSize, model.KernelSize),
	)
	bc1 := G.NewVector(g, T.Float32,
		G.WithName("bc1"),
		G.WithShape(model.Conv1Filters),
	)

	wfc1 := G.NewMatrix(g, T.Float32,
		G.WithName("wfc1"),
		G.WithShape(model.FlattenSize, model.HiddenSize),
	)
	bfc1 := G.NewVector(g, T.Float32,
		G.WithName("bfc1"),
		G.WithShape(model.HiddenSize),
	)

	wfc2 := G.NewMatrix(g, T.Float32,
		G.WithName("wfc2"),
		G.WithShape(model.HiddenSize, model.NumClasses),
	)
	bfc2 := G.NewVector(g, T.Float32,
		G.WithName("bfc2"),
		G.WithShape(model.NumClasses),
	)

	// Forward: Conv(1x1,pad=0) -> ReLU -> MaxPool(2x2) -> Flatten -> FC -> ReLU -> FC -> SoftMax
	conv, err := G.Conv2d(
		x, wc1,
		[]int{1, 1}, // stride
		[]int{0, 0}, // pad = 0  (important!)
		[]int{1, 1}, // dilation
		[]int{1, 1}, // groups
	)
	if err != nil {
		log.Fatal(err)
	}

	convB, err := G.BroadcastAdd(conv, bc1, nil, []byte{0, 2, 3})
	if err != nil {
		log.Fatal(err)
	}
	l1 := G.Must(G.Rectify(convB))

	// MaxPool 2x2, stride 2: 28x28 -> 14x14
	pool, err := G.MaxPool2D(l1,
		[]int{2, 2}, // kernel
		[]int{0, 0}, // pad
		[]int{2, 2}, // stride
	)
	if err != nil {
		log.Fatal(err)
	}

	// Flatten [1, C, H, W] -> [1, FlattenSize]
	flat := G.Must(G.Reshape(pool, T.Shape{1, model.FlattenSize}))

	// FC1: flat -> hidden, with bias broadcast
	fc1Lin := G.Must(G.Mul(flat, wfc1)) // [1, HiddenSize]
	fc1Add, err := G.BroadcastAdd(fc1Lin, bfc1, nil, []byte{0})
	if err != nil {
		log.Fatal(err)
	}
	fc1 := fc1Add
	fc1Act := G.Must(G.Rectify(fc1))

	// FC2: hidden -> logits, with bias broadcast
	fc2Lin := G.Must(G.Mul(fc1Act, wfc2)) // [1, NumClasses]
	logitsAdd, err := G.BroadcastAdd(fc2Lin, bfc2, nil, []byte{0})
	if err != nil {
		log.Fatal(err)
	}
	logits := logitsAdd

	probs := G.Must(G.SoftMax(logits)) // [1, NumClasses]

	// Assign trained weights
	if err := model.AssignWeights(mw, wc1, bc1, wfc1, bfc1, wfc2, bfc2); err != nil {
		log.Fatal(err)
	}

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}

		if len(req.Pixels) != model.InputHeight*model.InputWidth {
			http.Error(w, "pixels must have length 784 (28x28)", http.StatusBadRequest)
			return
		}

		// reshape to [1,1,28,28]
		xTensor := T.New(
			T.WithBacking(req.Pixels),
			T.WithShape(1, model.InputChannels, model.InputHeight, model.InputWidth),
		)
		G.Let(x, xTensor)

		if err := vm.RunAll(); err != nil {
			http.Error(w, "inference error: "+err.Error(), http.StatusInternalServerError)
			vm.Reset()
			return
		}

		raw := probs.Value().Data().([]float32) // length NumClasses
		vm.Reset()

		// argmax
		bestIdx := 0
		bestVal := raw[0]
		for i := 1; i < len(raw); i++ {
			if raw[i] > bestVal {
				bestVal = raw[i]
				bestIdx = i
			}
		}

		resp := PredictResponse{
			Digit: bestIdx,
			Probs: raw,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	addr := ":8080"
	fmt.Printf("Inference server listening on %s\n", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
