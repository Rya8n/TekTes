// install tensorflow go first!!!
package main

import (
	"fmt"
	"log"
)

func main() {

	model, err := tensorflow.LoadSavedModel("path/to/saved_model_dir", []string{"serve"}, nil)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Session.Close()

	input, err := tensorflow.NewTensor([][]float32{{1.0, 2.0, 3.0}}) // Dummmy input. Refer to the documentation for the model's input shape.
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}

	output, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("input").Output(0): input,
		},
		[]tensorflow.Output{
			model.Graph.Operation("output").Output(0),
		},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to run the session: %v", err)
	}

	fmt.Printf("Model output: %v\n", output[0].Value())
}
