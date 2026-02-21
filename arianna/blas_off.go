//go:build !blas

package arianna

// Stub: no BLAS acceleration. Pure Go fallback.
// Build with -tags blas to enable hardware acceleration.

var useBLAS = false

func blasMatVecF32(out []float32, mat []float32, vec []float32, rows, cols int) {}
func blasDotF32(a, b []float32) float32 { return 0 }
