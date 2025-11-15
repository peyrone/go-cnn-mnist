package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	mnist "github.com/petar/GoMNIST"
	"github.com/peyrone/go-cnn-mnist/internal/model"
	G "gorgonia.org/gorgonia"
	T "gorgonia.org/tensor"
)

const (
	seedNumber = 42
	batchSize  = 64
	maxEpochs  = 50
	learnRate  = 0.0001           // 1e-4, smaller LR for stability
	patience   = 5                // stop if no val improvement for N epochs
	modelPath  = "best_model.gob" // saved model file
	improveEps = 1e-4             // minimum val loss improvement
)

// Must stops the program if error occurs
func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// Encode a label as a one-hot vector
func oneHot(idx byte, classes int) []float32 {
	v := make([]float32, classes)
	v[int(idx)] = 1
	return v
}

// Convert MNIST images to a [N,1,28,28] tensor with values in [0,1]
func toTensorX(imgs []mnist.RawImage) *T.Dense {
	n := len(imgs)
	backing := make([]float32, 0, n*model.InputHeight*model.InputWidth)

	for _, im := range imgs {
		for _, p := range im {
			backing = append(backing, float32(p)/255.0)
		}
	}

	return T.New(
		T.WithBacking(backing),
		T.WithShape(n, model.InputChannels, model.InputHeight, model.InputWidth),
	)
}

// Convert labels to a [N,NumClasses] one-hot tensor
func toTensorY(labels []mnist.Label) *T.Dense {
	n := len(labels)
	backing := make([]float32, 0, n*model.NumClasses)

	for _, lb := range labels {
		backing = append(backing, oneHot(byte(lb), model.NumClasses)...)
	}

	return T.New(
		T.WithBacking(backing),
		T.WithShape(n, model.NumClasses),
	)
}

// Read a scalar float32 value from a Gorgonia node
func scalar32(n *G.Node) float32 {
	return n.Value().Data().(float32)
}

func main() {
	// Proper seeding
	rand.New(rand.NewSource(seedNumber))

	// Load dataset
	trainSet, testSet, err := mnist.Load("data/mnist")
	must(err)
	fmt.Printf("Loaded MNIST: %d train images\n", len(trainSet.Images))

	// Train/val split (90% train, 10% val)
	total := len(trainSet.Images)
	valSize := total / 10
	trainSize := total - valSize

	trainImgs := trainSet.Images[:trainSize]
	trainLabs := trainSet.Labels[:trainSize]
	valImgs := trainSet.Images[trainSize:]
	valLabs := trainSet.Labels[trainSize:]

	// Test set
	testImgs := testSet.Images
	testLabs := testSet.Labels

	fmt.Printf("Train: %d | Val: %d | Test: %d\n",
		len(trainImgs), len(valImgs), len(testImgs))

	// Build graph
	g := G.NewGraph()

	// Input placeholders (fixed batch size)
	x := G.NewTensor(g, T.Float32, 4,
		G.WithName("x"),
		G.WithShape(batchSize, model.InputChannels, model.InputHeight, model.InputWidth),
	)
	y := G.NewMatrix(g, T.Float32,
		G.WithName("y"),
		G.WithShape(batchSize, model.NumClasses),
	)

	// Parameters
	wc1 := G.NewTensor(g, T.Float32, 4,
		G.WithName("wc1"),
		G.WithShape(model.Conv1Filters, model.InputChannels, model.KernelSize, model.KernelSize),
		G.WithInit(G.GlorotN(1.0)),
	)
	bc1 := G.NewVector(g, T.Float32,
		G.WithName("bc1"),
		G.WithShape(model.Conv1Filters),
		G.WithInit(G.Zeroes()),
	)

	wfc1 := G.NewMatrix(g, T.Float32,
		G.WithName("wfc1"),
		G.WithShape(model.FlattenSize, model.HiddenSize),
		G.WithInit(G.GlorotN(1.0)),
	)
	bfc1 := G.NewVector(g, T.Float32,
		G.WithName("bfc1"),
		G.WithShape(model.HiddenSize),
		G.WithInit(G.Zeroes()),
	)

	wfc2 := G.NewMatrix(g, T.Float32,
		G.WithName("wfc2"),
		G.WithShape(model.HiddenSize, model.NumClasses),
		G.WithInit(G.GlorotN(1.0)),
	)
	bfc2 := G.NewVector(g, T.Float32,
		G.WithName("bfc2"),
		G.WithShape(model.NumClasses),
		G.WithInit(G.Zeroes()),
	)

	// Forward: Conv -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC -> LogSoftMax

	// IMPORTANT: kernel = 1x1, pad = 0 -> output stays 28x28
	conv, err := G.Conv2d(
		x, wc1,
		[]int{1, 1}, // stride
		[]int{0, 0}, // pad  (must be 0 to keep 28x28 for 1x1 conv)
		[]int{1, 1}, // dilation
		[]int{1, 1}, // groups
	)
	must(err)

	convB, err := G.BroadcastAdd(conv, bc1, nil, []byte{0, 2, 3})
	must(err)

	l1 := G.Must(G.Rectify(convB))

	// MaxPool 2x2 -> 28x28 -> 14x14
	pool, err := G.MaxPool2D(
		l1,
		[]int{2, 2}, // kernel
		[]int{0, 0}, // pad
		[]int{2, 2}, // stride
	)
	must(err)

	// Flatten [batchSize, C, H, W] -> [batchSize, FlattenSize]
	// With Conv1Filters=8, Kernel=1x1, pool 2x2:
	// C=8, H=W=14 → FlattenSize = 8*14*14 = 1568
	flat := G.Must(G.Reshape(pool, T.Shape{batchSize, model.FlattenSize}))

	// FC1: flat -> hidden
	fc1Lin := G.Must(G.Mul(flat, wfc1)) // (64, 128)
	fc1Add, err := G.BroadcastAdd(fc1Lin, bfc1, nil, []byte{0})
	must(err) // broadcast bfc1 over batch dimension
	fc1 := fc1Add
	fc1Act := G.Must(G.Rectify(fc1))

	// FC2: hidden -> logits
	fc2Lin := G.Must(G.Mul(fc1Act, wfc2)) // (64, 10)
	logitsAdd, err := G.BroadcastAdd(fc2Lin, bfc2, nil, []byte{0})
	must(err) // broadcast bfc2 over batch dimension
	logits := logitsAdd

	// LogSoftMax for NLL loss
	logProbs := G.Must(G.LogSoftMax(logits))

	// NLL loss: -mean(sum(y * logProbs))
	nllPerRow := G.Must(G.HadamardProd(y, G.Must(G.Neg(logProbs))))
	sum := G.Must(G.Sum(nllPerRow, 1)) // [batchSize]
	loss := G.Must(G.Mean(sum))        // scalar

	// Gradients
	params := G.Nodes{wc1, bc1, wfc1, bfc1, wfc2, bfc2}
	_, err = G.Grad(loss, params...)
	must(err)

	// VM + optimizer
	vm := G.NewTapeMachine(g, G.BindDualValues(params...))
	defer vm.Close()

	optim := G.NewAdamSolver(G.WithLearnRate(learnRate))

	// Training loop with early stopping
	trainBatches := len(trainImgs) / batchSize
	valBatches := len(valImgs) / batchSize

	bestValLoss := float32(1e9)
	noImprovement := 0

	fmt.Println("Start training...")

	for epoch := 1; epoch <= maxEpochs; epoch++ {
		epochStart := time.Now()

		// Shuffle training set
		rand.Shuffle(len(trainImgs), func(i, j int) {
			trainImgs[i], trainImgs[j] = trainImgs[j], trainImgs[i]
			trainLabs[i], trainLabs[j] = trainLabs[j], trainLabs[i]
		})

		var trainLossSum float32

		// Training
		for b := 0; b < trainBatches; b++ {
			start := b * batchSize
			end := start + batchSize

			xBatch := toTensorX(trainImgs[start:end])
			yBatch := toTensorY(trainLabs[start:end])

			G.Let(x, xBatch)
			G.Let(y, yBatch)

			must(vm.RunAll())

			if math.IsNaN(float64(scalar32(loss))) || math.IsInf(float64(scalar32(loss)), 0) {
				log.Fatalf("NaN/Inf loss detected at epoch %d batch %d", epoch, b+1)
			}

			must(optim.Step(G.NodesToValueGrads(params)))

			trainLossSum += scalar32(loss)
			vm.Reset()

			// Periodic mini-batch log
			if (b+1)%200 == 0 {
				fmt.Printf("  epoch %2d | batch %4d/%4d | loss=%.4f\n",
					epoch, b+1, trainBatches, scalar32(loss))
			}
		}
		avgTrainLoss := trainLossSum / float32(trainBatches)

		// Validation
		var valLossSum float32
		for b := 0; b < valBatches; b++ {
			start := b * batchSize
			end := start + batchSize

			xBatch := toTensorX(valImgs[start:end])
			yBatch := toTensorY(valLabs[start:end])

			G.Let(x, xBatch)
			G.Let(y, yBatch)

			must(vm.RunAll())
			valLossSum += scalar32(loss)
			vm.Reset()
		}
		avgValLoss := valLossSum / float32(valBatches)

		fmt.Printf("==> epoch %2d | trainLoss=%.4f | valLoss=%.4f | time=%s\n",
			epoch, avgTrainLoss, avgValLoss, time.Since(epochStart).Truncate(time.Millisecond))

		// Early stopping and save best model
		if avgValLoss < bestValLoss-improveEps {
			bestValLoss = avgValLoss
			noImprovement = 0

			mw := model.CaptureWeights(wc1, bc1, wfc1, bfc1, wfc2, bfc2)
			if err := model.Save(modelPath, mw); err != nil {
				log.Printf("WARN: cannot save model: %v\n", err)
			} else {
				fmt.Printf("  -> saved new best model to %s (valLoss=%.4f)\n",
					modelPath, bestValLoss)
			}
		} else {
			noImprovement++
			fmt.Printf("  no improvement for %d epoch(s)\n", noImprovement)
			if noImprovement >= patience {
				fmt.Println("  early stopping triggered")
				break
			}
		}
	}

	fmt.Println("Training finished.")
	fmt.Printf("Best validation loss: %.4f (saved at %s)\n", bestValLoss, modelPath)

	// Evaluate on test set
	fmt.Println("Evaluating on test set...")

	testBatches := len(testImgs) / batchSize
	var testLossSum float32
	var testCorrect int
	testTotal := testBatches * batchSize

	for b := 0; b < testBatches; b++ {
		start := b * batchSize
		end := start + batchSize

		xBatch := toTensorX(testImgs[start:end])
		yBatch := toTensorY(testLabs[start:end])

		G.Let(x, xBatch)
		G.Let(y, yBatch)

		must(vm.RunAll())

		// accumulate loss
		testLossSum += scalar32(loss)

		// compute accuracy from logProbs
		raw := logProbs.Value().Data().([]float32) // [batchSize * NumClasses]
		for i := 0; i < batchSize; i++ {
			offset := i * model.NumClasses
			bestIdx := 0
			bestVal := raw[offset]
			for c := 1; c < model.NumClasses; c++ {
				if raw[offset+c] > bestVal {
					bestVal = raw[offset+c]
					bestIdx = c
				}
			}
			if bestIdx == int(testLabs[start+i]) {
				testCorrect++
			}
		}

		vm.Reset()
	}

	avgTestLoss := testLossSum / float32(testBatches)
	testAcc := float32(testCorrect) / float32(testTotal)

	fmt.Printf("Test loss: %.4f | Test accuracy: %.4f\n", avgTestLoss, testAcc)
}
