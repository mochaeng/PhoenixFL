package torchbidings

// #cgo LDFLAGS: -L./build -lclassifier -ltorch -ltorch_cpu -ltorch_cuda -lcudart -lc10
// #cgo CXXFLAGS: -std=c++17
// #cgo CFLAGS: -D_GLIBCXX_USE_CXX11_ABI=1
// #include <stdio.h>
// #include <stdlib.h>
// #include "classifier.h"
import "C"
import (
	"errors"
	"unsafe"
)

type Classifier struct {
	model C.mModel
}

func NewModel(modelFile string) (*Classifier, error) {
	// LDFLAGS: -lstdc++ -L/usr/local/libtorch/lib -ltorch -ltorch_cpu -ltorch_cuda -lcudart -lc10
	// CXXFLAGS: -std=c++17 -I${SRCDIR} -g -O3
	cModelFile := C.CString(modelFile)
	defer C.free(unsafe.Pointer(cModelFile))

	model := C.NewModel(cModelFile)
	if model == nil {
		return nil, errors.New("failed to load model")
	}
	return &Classifier{model: model}, nil
}

func (m *Classifier) PredictIsPositiveBinary(inputData []float32) (bool, error) {
	if m.model == nil {
		return false, errors.New("model not initialized")
	}
	numFeatures := C.int(len(inputData))
	result := C.PredictIsPositiveBinary(m.model, (*C.float)(unsafe.Pointer(&inputData[0])), numFeatures)
	if result == -1 {
		return false, errors.New("prediction failed")
	}
	return result == 1, nil
}

func (m *Classifier) Delete() {
	if m.model != nil {
		C.DeleteModel(m.model)
		m.model = nil
	}
}
