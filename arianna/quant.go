package arianna

import (
	"math"
	"runtime"
	"sync"
)

// Number of goroutines for parallel matmul
var numWorkers = runtime.NumCPU()

// Quantized matrix-vector multiply: out[rows] = Q4_0_mat[rows, cols] × vec[cols]
// mat is raw Q4_0 bytes: each row = (cols/32) blocks, each block = 18 bytes
// Parallelized across rows using goroutines
func matVecQ4_0(out []float32, mat []byte, vec []float32, rows, cols int) {
	blocksPerRow := cols / 32
	bytesPerRow := blocksPerRow * Q4_0_BYTES_PER_BLOCK

	if rows < numWorkers*4 {
		// Small matrix — single thread
		matVecQ4_0Range(out, mat, vec, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matVecQ4_0Range(out, mat, vec, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matVecQ4_0Range(out []float32, mat []byte, vec []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			boff := rowOff + b*Q4_0_BYTES_PER_BLOCK
			scale := float16to32(uint16(mat[boff]) | uint16(mat[boff+1])<<8)
			vecOff := b * 32

			for j := 0; j < 16; j++ {
				qbyte := mat[boff+2+j]
				lo := float32(int(qbyte&0x0F) - 8)
				hi := float32(int(qbyte>>4) - 8)

				sum += scale * lo * vec[vecOff+j]
				sum += scale * hi * vec[vecOff+16+j]
			}
		}
		out[i] = sum
	}
}

// Quantized matrix-vector multiply for Q8_0
// Parallelized across rows using goroutines
func matVecQ8_0(out []float32, mat []byte, vec []float32, rows, cols int) {
	blocksPerRow := cols / 32
	bytesPerRow := blocksPerRow * Q8_0_BYTES_PER_BLOCK

	if rows < numWorkers*4 {
		matVecQ8_0Range(out, mat, vec, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matVecQ8_0Range(out, mat, vec, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matVecQ8_0Range(out []float32, mat []byte, vec []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)

		for b := 0; b < blocksPerRow; b++ {
			boff := rowOff + b*Q8_0_BYTES_PER_BLOCK
			scale := float16to32(uint16(mat[boff]) | uint16(mat[boff+1])<<8)
			vecOff := b * 32

			for j := 0; j < 32; j++ {
				q := int8(mat[boff+2+j])
				sum += scale * float32(q) * vec[vecOff+j]
			}
		}
		out[i] = sum
	}
}

// F32 matrix-vector multiply (for norm weights etc)
func matVecF32(out []float32, mat []float32, vec []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		sum := float32(0)
		off := i * cols
		for j := 0; j < cols; j++ {
			sum += mat[off+j] * vec[j]
		}
		out[i] = sum
	}
}

// Dot product
func dotF32(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Fast exp approximation (Schraudolph's method, ~1.5% max error)
// Avoids float64 round-trip through math.Exp
func fastExp(x float32) float32 {
	if x < -88 {
		return 0
	}
	if x > 88 {
		return math.MaxFloat32
	}
	// 2^23 / ln(2) ≈ 12102203.16, bias = 127 * 2^23 = 1065353216
	i := int32(x*12102203.0) + 1065353216
	return math.Float32frombits(uint32(i))
}

// RMSNorm: out = (x / rms(x)) * weight
func rmsNormF32(out, x, weight []float32, eps float32) {
	n := len(x)
	ss := float32(0)
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * ss * weight[i]
	}
}

// RoPE precomputed cos/sin table
type RoPETable struct {
	Cos []float32 // [contextLen * headDim/2]
	Sin []float32 // [contextLen * headDim/2]
	HalfDim int
}

func NewRoPETable(contextLen, headDim int, theta float32) *RoPETable {
	halfDim := headDim / 2
	cos := make([]float32, contextLen*halfDim)
	sin := make([]float32, contextLen*halfDim)

	for pos := 0; pos < contextLen; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / float32(math.Pow(float64(theta), float64(2*i)/float64(headDim)))
			angle := float64(float32(pos) * freq)
			cos[pos*halfDim+i] = float32(math.Cos(angle))
			sin[pos*halfDim+i] = float32(math.Sin(angle))
		}
	}
	return &RoPETable{Cos: cos, Sin: sin, HalfDim: halfDim}
}

// Apply RoPE using precomputed table
func ropeF32(q, k []float32, pos int, rope *RoPETable, nHeads, nKVHeads, headDim int) {
	halfDim := rope.HalfDim
	off := pos * halfDim

	for i := 0; i < halfDim; i++ {
		cos := rope.Cos[off+i]
		sin := rope.Sin[off+i]

		for h := 0; h < nHeads; h++ {
			hoff := h*headDim + i*2
			q0, q1 := q[hoff], q[hoff+1]
			q[hoff] = q0*cos - q1*sin
			q[hoff+1] = q0*sin + q1*cos
		}

		for h := 0; h < nKVHeads; h++ {
			hoff := h*headDim + i*2
			k0, k1 := k[hoff], k[hoff+1]
			k[hoff] = k0*cos - k1*sin
			k[hoff+1] = k0*sin + k1*cos
		}
	}
}

// Softmax in-place over first n elements
func softmaxF32(x []float32, n int) {
	maxVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxVal {
			maxVal = x[i]
		}
	}
	sum := float32(0)
	for i := 0; i < n; i++ {
		x[i] = fastExp(x[i] - maxVal)
		sum += x[i]
	}
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}
