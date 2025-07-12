/*
tensor.go GS 7/11/2025

this could be annoying since we do not have access to OOP abstractions
but to be honest, those were built from scratch to make C++ from C
so really how hard can it be

tensor dimensions are the easiest since in every other implementation they are
CONTIGIOUS. all we have to do is allocate the memory block and partition it based
on the dimension value
	- we will default to 2 dimensions

*/

package tensor

type DType int 

const (
	Float32 DType = iota
	Float16
	Int8
)

type Tensor struct {
	Data []float32
	Shape []int // example [2, 3, 4]
	Strides []int // cached strides for index computation
	Offset int // for slicing
	Dtype Dtype // float32, float16 ...
	Device string // "cpu" "gpu" "simd-cpu"
}

// helper functions
func numElements(shape []int) int {
	// return the number of elements needed based on the shape array
	// (remember we are using flattened 1d array representation)
	// syntax NOTE: "range" gives all elements in arg (in this case, shape array)
	size := 1
	for _, dim := range shape { 
		size *= dim
	}
	return size
}

func computeStrides(shape []int) []int {
	/*
	Takes the input shape array (slice?) and returns the "strides" needed to move along each axis as an array
	ex: [2, 3, 4] means "2 blocks of 3 by 4 matrices"
		stored in a single memory block, this means the first of the outermost 2 blocks
		will start at 0 but the second will start AFTER the 3 by 4 matrix aka AFTER 3 * 4 = 12 items
	*/

	n := len(shape)
	strides := make([]int, n) // syntax NOTE: "make" allocates zeroed int slice of length n

	strides[n-1] = 1
	for i := n - 2; i >= 0; i-- {
		strides[i] = shape[i+1] * strides[i+1]
	}
	return strides
}




// construction
// syntax NOTE: 
//     we are usually using pointers, {...} is a composite literal, & gets pointer to that value
func Zeros(shape ...int) *Tensor {
	// fill Data with 0s

	// get number of elements to create:
	size := numElements(shape)
	data := make([]float32, size)
	return &Tensor {
		Data: data,
		Shape: shape,
		Strides: computeStrides(shape),
		Offset: 0,
		Dtype: Float32,
		Device: "cpu",
	}
}

func Ones(shape ...int) *Tensor
func Empty(shape ...int) *Tensor
func Random(shape ...int) *Tensor
func FromData(data []float32, shape []int) (*Tensor, error)

// access
func (t *Tensor) NumDims() int
func (t *Tensor) Size() int
func (t *Tensor) Strides() []int
func (t *Tensor) At(indices ...int) float32
func (t *Tensor) Clone() *Tensor 
func (t *Tensor) Reshape(newShape []int) error

// slicing
func (t *Tensor) Slice(axis int, index int) (*Tensor, error)
func (t *Tensor) Permute(axes []int) (*Tensor, error)
func (t *Tensor) Transpose() *Tensor // Shortcut for 2D only

// elementwise ops
func (a *Tensor) Add(b *Tensor) (*Tensor, error)
func (a *Tensor) Sub(b *Tensor) (*Tensor, error)
func (a *Tensor) Mul(b *Tensor) (*Tensor, error)
func (a *Tensor) Div(b *Tensor) (*Tensor, error)
func (t *Tensor) Apply(fn func(float32) float32) *Tensor

// math operations
func (a *Tensor) MatMul(b *Tensor) (*Tensor, error) // Only works for 2D (or batched matmul later)
func (t *Tensor) Softmax(axis int) (*Tensor, error)
func (t *Tensor) Sum(axis int) (*Tensor, error)
func (t *Tensor) Mean(axis int) (*Tensor, error)

// utils
func (t *Tensor) Print(prefix string)
func (t *Tensor) Equals(b *Tensor) bool

