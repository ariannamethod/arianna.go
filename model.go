package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

type Config struct {
	Dim        int
	HiddenDim  int
	NLayers    int
	NHeads     int
	NKVHeads   int
	VocabSize  int
	ContextLen int
	HeadDim    int
	KVGroups   int
	RopeTheta  float32
	NormEps    float32
}

// Quantized tensor: raw bytes + metadata
type QTensor struct {
	Data  []byte
	Type  uint32
	Rows  int
	Cols  int
}

// Dispatch matmul based on quantization type
func (qt *QTensor) MatVec(out []float32, vec []float32) {
	switch qt.Type {
	case GGML_TYPE_Q4_0:
		matVecQ4_0(out, qt.Data, vec, qt.Rows, qt.Cols)
	case GGML_TYPE_Q8_0:
		matVecQ8_0(out, qt.Data, vec, qt.Rows, qt.Cols)
	}
}

// All model weights — kept quantized
type Weights struct {
	TokenEmbd *QTensor   // [vocab_size, dim]
	Output    *QTensor   // [vocab_size, dim]
	OutputNorm []float32 // [dim] — always f32

	// Per-layer
	AttnNorm [][]float32 // [n_layers][dim] — always f32
	WQ       []*QTensor  // [n_layers]
	WK       []*QTensor
	WV       []*QTensor
	WO       []*QTensor
	FFNNorm  [][]float32 // [n_layers][dim] — always f32
	FFNGate  []*QTensor
	FFNUp    []*QTensor
	FFNDown  []*QTensor
}

type RunState struct {
	X      []float32
	XB     []float32
	XB2    []float32
	HB     []float32
	HB2    []float32
	Q      []float32
	K      []float32
	V      []float32
	Att    []float32
	Logits []float32
	KeyCache   []float32
	ValueCache []float32
}

type Model struct {
	Config  Config
	Weights Weights
	State   RunState
	Tok     *Tokenizer
}

// Read raw tensor bytes from GGUF without dequantizing
func readRawTensor(g *GGUFFile, name string) (*QTensor, error) {
	var info *GGUFTensorInfo
	for i := range g.Tensors {
		if g.Tensors[i].Name == name {
			info = &g.Tensors[i]
			break
		}
	}
	if info == nil {
		return nil, fmt.Errorf("tensor %q not found", name)
	}

	nelems := uint64(1)
	for _, s := range info.Shape {
		nelems *= s
	}

	// Calculate byte size based on type
	var nbytes uint64
	switch info.Type {
	case GGML_TYPE_F32:
		nbytes = nelems * 4
	case GGML_TYPE_F16:
		nbytes = nelems * 2
	case GGML_TYPE_Q4_0:
		nblocks := nelems / BLOCK_SIZE
		nbytes = nblocks * Q4_0_BYTES_PER_BLOCK
	case GGML_TYPE_Q8_0:
		nblocks := nelems / BLOCK_SIZE
		nbytes = nblocks * Q8_0_BYTES_PER_BLOCK
	default:
		return nil, fmt.Errorf("unsupported type %d for %s", info.Type, name)
	}

	g.file.Seek(g.DataOff+int64(info.Offset), 0)
	data := make([]byte, nbytes)
	if _, err := g.file.Read(data); err != nil {
		return nil, err
	}

	rows := int(info.Shape[len(info.Shape)-1]) // last dim = rows in GGUF (transposed)
	cols := int(info.Shape[0])
	if len(info.Shape) == 1 {
		rows = int(info.Shape[0])
		cols = 1
	}

	return &QTensor{
		Data: data,
		Type: info.Type,
		Rows: rows,
		Cols: cols,
	}, nil
}

// Read f32 tensor (for norms)
func readF32Tensor(g *GGUFFile, name string) ([]float32, error) {
	data, _, err := g.ReadTensorF32(name)
	return data, err
}

func LoadModel(g *GGUFFile) (*Model, error) {
	m := &Model{}

	m.Config = Config{
		Dim:        int(g.GetUint32("llama.embedding_length")),
		HiddenDim:  int(g.GetUint32("llama.feed_forward_length")),
		NLayers:    int(g.GetUint32("llama.block_count")),
		NHeads:     int(g.GetUint32("llama.attention.head_count")),
		NKVHeads:   int(g.GetUint32("llama.attention.head_count_kv")),
		VocabSize:  int(g.GetUint32("llama.vocab_size")),
		ContextLen: int(g.GetUint32("llama.context_length")),
		RopeTheta:  g.GetFloat32("llama.rope.freq_base"),
		NormEps:    g.GetFloat32("llama.attention.layer_norm_rms_epsilon"),
	}
	c := &m.Config
	c.HeadDim = c.Dim / c.NHeads
	c.KVGroups = c.NHeads / c.NKVHeads
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000.0
	}
	if c.NormEps == 0 {
		c.NormEps = 1e-6
	}

	fmt.Printf("Model: dim=%d, layers=%d, heads=%d, kv_heads=%d, ffn=%d, vocab=%d, ctx=%d\n",
		c.Dim, c.NLayers, c.NHeads, c.NKVHeads, c.HiddenDim, c.VocabSize, c.ContextLen)

	kvDim := c.NKVHeads * c.HeadDim

	// Load weights — quantized tensors stay as raw bytes
	fmt.Println("Loading weights (quantized)...")

	var err error

	loadQ := func(name string) *QTensor {
		if err != nil {
			return nil
		}
		qt, e := readRawTensor(g, name)
		if e != nil {
			err = e
			return nil
		}
		return qt
	}

	loadF32 := func(name string) []float32 {
		if err != nil {
			return nil
		}
		data, e := readF32Tensor(g, name)
		if e != nil {
			err = e
			return nil
		}
		return data
	}

	m.Weights.TokenEmbd = loadQ("token_embd.weight")
	m.Weights.Output = loadQ("output.weight")
	m.Weights.OutputNorm = loadF32("output_norm.weight")

	m.Weights.AttnNorm = make([][]float32, c.NLayers)
	m.Weights.WQ = make([]*QTensor, c.NLayers)
	m.Weights.WK = make([]*QTensor, c.NLayers)
	m.Weights.WV = make([]*QTensor, c.NLayers)
	m.Weights.WO = make([]*QTensor, c.NLayers)
	m.Weights.FFNNorm = make([][]float32, c.NLayers)
	m.Weights.FFNGate = make([]*QTensor, c.NLayers)
	m.Weights.FFNUp = make([]*QTensor, c.NLayers)
	m.Weights.FFNDown = make([]*QTensor, c.NLayers)

	for l := 0; l < c.NLayers; l++ {
		p := fmt.Sprintf("blk.%d", l)
		m.Weights.AttnNorm[l] = loadF32(p + ".attn_norm.weight")
		m.Weights.WQ[l] = loadQ(p + ".attn_q.weight")
		m.Weights.WK[l] = loadQ(p + ".attn_k.weight")
		m.Weights.WV[l] = loadQ(p + ".attn_v.weight")
		m.Weights.WO[l] = loadQ(p + ".attn_output.weight")
		m.Weights.FFNNorm[l] = loadF32(p + ".ffn_norm.weight")
		m.Weights.FFNGate[l] = loadQ(p + ".ffn_gate.weight")
		m.Weights.FFNUp[l] = loadQ(p + ".ffn_up.weight")
		m.Weights.FFNDown[l] = loadQ(p + ".ffn_down.weight")

		if err != nil {
			return nil, err
		}
		if (l+1)%5 == 0 || l == c.NLayers-1 {
			fmt.Printf("  Layer %d/%d\n", l+1, c.NLayers)
		}
	}
	if err != nil {
		return nil, err
	}

	// Allocate runtime state (only f32 buffers — small)
	m.State = RunState{
		X:          make([]float32, c.Dim),
		XB:         make([]float32, c.Dim),
		XB2:        make([]float32, c.Dim),
		HB:         make([]float32, c.HiddenDim),
		HB2:        make([]float32, c.HiddenDim),
		Q:          make([]float32, c.Dim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, c.NHeads*c.ContextLen),
		Logits:     make([]float32, c.VocabSize),
		KeyCache:   make([]float32, c.NLayers*c.ContextLen*kvDim),
		ValueCache: make([]float32, c.NLayers*c.ContextLen*kvDim),
	}

	// Load tokenizer
	m.Tok, err = LoadTokenizerFromGGUF(g)
	if err != nil {
		return nil, err
	}

	fmt.Println("Model ready.")
	return m, nil
}

// Dequantize a single row from token embedding (for embedding lookup)
func (m *Model) embedToken(token int) {
	qt := m.Weights.TokenEmbd
	dim := m.Config.Dim

	switch qt.Type {
	case GGML_TYPE_Q4_0:
		blocksPerRow := dim / 32
		rowOff := token * blocksPerRow * Q4_0_BYTES_PER_BLOCK
		for b := 0; b < blocksPerRow; b++ {
			boff := rowOff + b*Q4_0_BYTES_PER_BLOCK
			scale := float16to32(uint16(qt.Data[boff]) | uint16(qt.Data[boff+1])<<8)
			for j := 0; j < 16; j++ {
				qbyte := qt.Data[boff+2+j]
				m.State.X[b*32+j] = scale * float32(int(qbyte&0x0F)-8)
				m.State.X[b*32+16+j] = scale * float32(int(qbyte>>4)-8)
			}
		}
	case GGML_TYPE_Q8_0:
		blocksPerRow := dim / 32
		rowOff := token * blocksPerRow * Q8_0_BYTES_PER_BLOCK
		for b := 0; b < blocksPerRow; b++ {
			boff := rowOff + b*Q8_0_BYTES_PER_BLOCK
			scale := float16to32(uint16(qt.Data[boff]) | uint16(qt.Data[boff+1])<<8)
			for j := 0; j < 32; j++ {
				m.State.X[b*32+j] = scale * float32(int8(qt.Data[boff+2+j]))
			}
		}
	case GGML_TYPE_F16:
		off := token * dim * 2
		for i := 0; i < dim; i++ {
			m.State.X[i] = float16to32(uint16(qt.Data[off+i*2]) | uint16(qt.Data[off+i*2+1])<<8)
		}
	}
}

// Forward pass — single token at position pos
func (m *Model) Forward(token int, pos int) []float32 {
	c := &m.Config
	w := &m.Weights
	s := &m.State
	dim := c.Dim
	kvDim := c.NKVHeads * c.HeadDim
	headDim := c.HeadDim

	// Token embedding lookup
	m.embedToken(token)

	for l := 0; l < c.NLayers; l++ {
		// Pre-attention RMSNorm
		rmsNormF32(s.XB, s.X, w.AttnNorm[l], c.NormEps)

		// QKV projections (quantized matmul)
		w.WQ[l].MatVec(s.Q, s.XB)
		w.WK[l].MatVec(s.K, s.XB)
		w.WV[l].MatVec(s.V, s.XB)

		// RoPE
		ropeF32(s.Q, s.K, pos, headDim, c.NHeads, c.NKVHeads, c.RopeTheta)

		// Cache K,V
		kvOff := l*c.ContextLen*kvDim + pos*kvDim
		copy(s.KeyCache[kvOff:kvOff+kvDim], s.K)
		copy(s.ValueCache[kvOff:kvOff+kvDim], s.V)

		// Multi-head attention
		layerKVOff := l * c.ContextLen * kvDim
		for h := 0; h < c.NHeads; h++ {
			kvH := h / c.KVGroups
			qOff := h * headDim
			kvHOff := kvH * headDim

			for t := 0; t <= pos; t++ {
				kOff := layerKVOff + t*kvDim + kvHOff
				score := dotF32(s.Q[qOff:qOff+headDim], s.KeyCache[kOff:kOff+headDim])
				s.Att[h*c.ContextLen+t] = score / float32(math.Sqrt(float64(headDim)))
			}

			softmaxF32(s.Att[h*c.ContextLen:h*c.ContextLen+pos+1], pos+1)

			xbOff := qOff
			for d := 0; d < headDim; d++ {
				s.XB2[xbOff+d] = 0
			}
			for t := 0; t <= pos; t++ {
				a := s.Att[h*c.ContextLen+t]
				vOff := layerKVOff + t*kvDim + kvHOff
				for d := 0; d < headDim; d++ {
					s.XB2[xbOff+d] += a * s.ValueCache[vOff+d]
				}
			}
		}

		// Output projection + residual
		w.WO[l].MatVec(s.XB, s.XB2)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// Pre-FFN RMSNorm
		rmsNormF32(s.XB, s.X, w.FFNNorm[l], c.NormEps)

		// SwiGLU FFN (quantized matmul)
		w.FFNGate[l].MatVec(s.HB, s.XB)
		w.FFNUp[l].MatVec(s.HB2, s.XB)

		for i := 0; i < c.HiddenDim; i++ {
			s.HB[i] = (s.HB[i] / (1.0 + float32(math.Exp(float64(-s.HB[i]))))) * s.HB2[i]
		}

		w.FFNDown[l].MatVec(s.XB, s.HB)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// Final RMSNorm
	rmsNormF32(s.X, s.X, w.OutputNorm, c.NormEps)

	// LM head → logits (quantized matmul)
	w.Output.MatVec(s.Logits, s.X)

	return s.Logits
}

// Generate text
func (m *Model) Generate(prompt string, maxTokens int, temp float32, topP float32, repPenalty float32) string {
	ids := m.Tok.Encode(prompt)
	generated := make([]int, 0, maxTokens)

	fmt.Printf("  [%d prompt tokens] ", len(ids))

	// Process prompt
	for i, id := range ids {
		m.Forward(id, i)
	}
	pos := len(ids)

	// Sample first token
	logitsCopy := make([]float32, len(m.State.Logits))
	copy(logitsCopy, m.State.Logits)
	applyRepPenalty(logitsCopy, ids, repPenalty)
	next := sampleTopP(logitsCopy, temp, topP)
	generated = append(generated, next)
	fmt.Print(m.Tok.DecodeOne(next))

	// Autoregressive loop
	for i := 1; i < maxTokens; i++ {
		m.Forward(next, pos)
		pos++

		copy(logitsCopy, m.State.Logits)
		allToks := append(ids, generated...)
		applyRepPenalty(logitsCopy, allToks, repPenalty)
		next = sampleTopP(logitsCopy, temp, topP)

		if next == m.Tok.EOS {
			break
		}
		generated = append(generated, next)
		fmt.Print(m.Tok.DecodeOne(next))
	}
	fmt.Println()

	return m.Tok.Decode(generated)
}

func applyRepPenalty(logits []float32, prev []int, penalty float32) {
	if penalty == 1.0 {
		return
	}
	for _, tok := range prev {
		if tok >= 0 && tok < len(logits) {
			if logits[tok] > 0 {
				logits[tok] /= penalty
			} else {
				logits[tok] *= penalty
			}
		}
	}
}

func sampleTopP(logits []float32, temp, topP float32) int {
	n := len(logits)

	// Greedy (argmax) when temp=0
	if temp <= 0 {
		best := 0
		for i := 1; i < n; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}

	for i := 0; i < n; i++ {
		logits[i] /= temp
	}
	softmaxF32(logits, n)

	type tp struct {
		id int
		p  float32
	}
	items := make([]tp, n)
	for i := 0; i < n; i++ {
		items[i] = tp{i, logits[i]}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].p > items[j].p
	})

	cum := float32(0)
	cutoff := n
	for i := 0; i < n; i++ {
		cum += items[i].p
		if cum > topP {
			cutoff = i + 1
			break
		}
	}

	cum = 0
	for i := 0; i < cutoff; i++ {
		cum += items[i].p
	}
	r := rand.Float32() * cum
	cum = 0
	for i := 0; i < cutoff; i++ {
		cum += items[i].p
		if cum >= r {
			return items[i].id
		}
	}
	return items[0].id
}
