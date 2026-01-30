package arianna

import (
	"math"
)

// Quantized matrix-vector multiply: out[rows] = Q4_0_mat[rows, cols] × vec[cols]
// mat is raw Q4_0 bytes: each row = (cols/32) blocks, each block = 18 bytes
func matVecQ4_0(out []float32, mat []byte, vec []float32, rows, cols int) {
	blocksPerRow := cols / 32
	bytesPerRow := blocksPerRow * Q4_0_BYTES_PER_BLOCK

	for i := 0; i < rows; i++ {
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
// mat is raw Q8_0 bytes: each row = (cols/32) blocks, each block = 34 bytes
func matVecQ8_0(out []float32, mat []byte, vec []float32, rows, cols int) {
	blocksPerRow := cols / 32
	bytesPerRow := blocksPerRow * Q8_0_BYTES_PER_BLOCK

	for i := 0; i < rows; i++ {
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

// Apply RoPE to Q and K vectors in-place
func ropeF32(q, k []float32, pos, headDim, nHeads, nKVHeads int, theta float32) {
	for i := 0; i < headDim/2; i++ {
		freq := 1.0 / float32(math.Pow(float64(theta), float64(2*i)/float64(headDim)))
		angle := float32(pos) * freq
		cos := float32(math.Cos(float64(angle)))
		sin := float32(math.Sin(float64(angle)))

		for h := 0; h < nHeads; h++ {
			off := h*headDim + i*2
			q0, q1 := q[off], q[off+1]
			q[off] = q0*cos - q1*sin
			q[off+1] = q0*sin + q1*cos
		}

		for h := 0; h < nKVHeads; h++ {
			off := h*headDim + i*2
			k0, k1 := k[off], k[off+1]
			k[off] = k0*cos - k1*sin
			k[off+1] = k0*sin + k1*cos
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
		x[i] = float32(math.Exp(float64(x[i] - maxVal)))
		sum += x[i]
	}
	for i := 0; i < n; i++ {
		x[i] /= sum
	}
}
