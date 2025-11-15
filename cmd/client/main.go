package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/nfnt/resize"
	"github.com/peyrone/go-cnn-mnist/internal/model"
)

type PredictRequest struct {
	Pixels []float32 `json:"pixels"`
}

type PredictResponse struct {
	Digit int       `json:"digit"`
	Probs []float32 `json:"probs"`
}

func main() {
	const imgPath = "test/three.png"

	pixels, err := loadAndPreprocess(imgPath, model.InputWidth, model.InputHeight)
	if err != nil {
		log.Fatal("preprocess error:", err)
	}

	reqBody := PredictRequest{Pixels: pixels}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		log.Fatal("marshal error:", err)
	}

	resp, err := http.Post(
		"http://localhost:8080/predict",
		"application/json",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		log.Fatal("HTTP error:", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		log.Fatalf("Bad status: %s\n%s", resp.Status, body)
	}

	var pr PredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&pr); err != nil {
		log.Fatal("decode error:", err)
	}

	fmt.Printf("Predicted digit: %d\n", pr.Digit)
	fmt.Printf("Probs: %v\n", pr.Probs)
}

func loadAndPreprocess(path string, targetWidth, targetHeight int) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return nil, err
	}

	// Convert to grayscale using the standard model
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := color.GrayModel.Convert(img.At(x, y))
			gray.Set(x, y, c)
		}
	}

	// Resize to target size
	resized := resize.Resize(
		uint(targetWidth),
		uint(targetHeight),
		gray,
		resize.Lanczos3,
	)

	return imageToNormalizedPixels(resized), nil
}

func imageToNormalizedPixels(img image.Image) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	pixels := make([]float32, w*h)
	idx := 0

	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			// r is 16-bit in [0, 65535]; normalize to [0,1]
			pixels[idx] = float32(r) / 65535.0
			idx++
		}
	}

	return pixels
}
