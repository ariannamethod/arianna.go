package arianna

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// GGUF value types
const (
	GGUF_TYPE_UINT8   = 0
	GGUF_TYPE_INT8    = 1
	GGUF_TYPE_UINT16  = 2
	GGUF_TYPE_INT16   = 3
	GGUF_TYPE_UINT32  = 4
	GGUF_TYPE_INT32   = 5
	GGUF_TYPE_FLOAT32 = 6
	GGUF_TYPE_BOOL    = 7
	GGUF_TYPE_STRING  = 8
	GGUF_TYPE_ARRAY   = 9
	GGUF_TYPE_UINT64  = 10
	GGUF_TYPE_INT64   = 11
	GGUF_TYPE_FLOAT64 = 12
)

// GGUF tensor data types
const (
	GGML_TYPE_F32  = 0
	GGML_TYPE_F16  = 1
	GGML_TYPE_Q4_0 = 2
	GGML_TYPE_Q4_1 = 3
	GGML_TYPE_Q5_0 = 6
	GGML_TYPE_Q5_1 = 7
	GGML_TYPE_Q8_0 = 8
	GGML_TYPE_Q8_1 = 9
)

// Block size for quantized types (weights per block)
const BLOCK_SIZE = 32

// Q4_0 block: 2 bytes (float16 scale) + 16 bytes (32 x 4-bit quants) = 18 bytes
const Q4_0_BYTES_PER_BLOCK = 18

// Q8_0 block: 2 bytes (float16 scale) + 32 bytes (32 x int8 quants) = 34 bytes
const Q8_0_BYTES_PER_BLOCK = 34

type GGUFHeader struct {
	Magic         [4]byte
	Version       uint32
	TensorCount   uint64
	MetadataCount uint64
}

type GGUFTensorInfo struct {
	Name   string
	NDims  uint32
	Shape  []uint64
	Type   uint32
	Offset uint64
}

type GGUFFile struct {
	Header   GGUFHeader
	Metadata map[string]interface{}
	Tensors  []GGUFTensorInfo
	DataOff  int64 // offset where tensor data begins
	file     *os.File
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype uint32) (interface{}, error) {
	switch vtype {
	case GGUF_TYPE_UINT8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_INT8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_UINT16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_INT16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_UINT32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_INT32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_FLOAT32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_BOOL:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case GGUF_TYPE_STRING:
		return readString(r)
	case GGUF_TYPE_UINT64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_INT64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_FLOAT64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case GGUF_TYPE_ARRAY:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			v, err := readValue(r, elemType)
			if err != nil {
				return nil, fmt.Errorf("array element %d: %w", i, err)
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", vtype)
	}
}

func OpenGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	g := &GGUFFile{
		file:     f,
		Metadata: make(map[string]interface{}),
	}

	// Read header
	if err := binary.Read(f, binary.LittleEndian, &g.Header.Magic); err != nil {
		return nil, err
	}
	if string(g.Header.Magic[:]) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file (magic: %q)", g.Header.Magic)
	}
	if err := binary.Read(f, binary.LittleEndian, &g.Header.Version); err != nil {
		return nil, err
	}

	// Version 3: uint64 for counts
	if g.Header.Version >= 3 {
		if err := binary.Read(f, binary.LittleEndian, &g.Header.TensorCount); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, &g.Header.MetadataCount); err != nil {
			return nil, err
		}
	} else {
		var tc, mc uint32
		binary.Read(f, binary.LittleEndian, &tc)
		binary.Read(f, binary.LittleEndian, &mc)
		g.Header.TensorCount = uint64(tc)
		g.Header.MetadataCount = uint64(mc)
	}

	fmt.Printf("GGUF v%d: %d tensors, %d metadata entries\n",
		g.Header.Version, g.Header.TensorCount, g.Header.MetadataCount)

	// Read metadata
	for i := uint64(0); i < g.Header.MetadataCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("metadata key %d: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, fmt.Errorf("metadata type %d: %w", i, err)
		}
		val, err := readValue(f, vtype)
		if err != nil {
			return nil, fmt.Errorf("metadata value %q: %w", key, err)
		}
		g.Metadata[key] = val
	}

	// Read tensor infos
	g.Tensors = make([]GGUFTensorInfo, g.Header.TensorCount)
	for i := uint64(0); i < g.Header.TensorCount; i++ {
		name, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("tensor name %d: %w", i, err)
		}
		var ndims uint32
		if err := binary.Read(f, binary.LittleEndian, &ndims); err != nil {
			return nil, err
		}
		shape := make([]uint64, ndims)
		for d := uint32(0); d < ndims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &shape[d]); err != nil {
				return nil, err
			}
		}
		var dtype uint32
		if err := binary.Read(f, binary.LittleEndian, &dtype); err != nil {
			return nil, err
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, err
		}
		g.Tensors[i] = GGUFTensorInfo{
			Name:   name,
			NDims:  ndims,
			Shape:  shape,
			Type:   dtype,
			Offset: offset,
		}
	}

	// Data starts at next 32-byte aligned offset after tensor info
	pos, _ := f.Seek(0, io.SeekCurrent)
	g.DataOff = align(pos, 32)

	return g, nil
}

func (g *GGUFFile) Close() {
	g.file.Close()
}

// GetUint32 returns a uint32 metadata value
func (g *GGUFFile) GetUint32(key string) uint32 {
	v, ok := g.Metadata[key]
	if !ok {
		return 0
	}
	switch val := v.(type) {
	case uint32:
		return val
	case uint64:
		return uint32(val)
	case int32:
		return uint32(val)
	default:
		return 0
	}
}

// GetFloat32 returns a float32 metadata value
func (g *GGUFFile) GetFloat32(key string) float32 {
	v, ok := g.Metadata[key]
	if !ok {
		return 0
	}
	switch val := v.(type) {
	case float32:
		return val
	case float64:
		return float32(val)
	default:
		return 0
	}
}

// GetString returns a string metadata value
func (g *GGUFFile) GetString(key string) string {
	v, ok := g.Metadata[key]
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

// GetStringArray returns a string array metadata value
func (g *GGUFFile) GetStringArray(key string) []string {
	v, ok := g.Metadata[key]
	if !ok {
		return nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	strs := make([]string, len(arr))
	for i, a := range arr {
		strs[i], _ = a.(string)
	}
	return strs
}

// GetFloat32Array returns a float32 array metadata value
func (g *GGUFFile) GetFloat32Array(key string) []float32 {
	v, ok := g.Metadata[key]
	if !ok {
		return nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	floats := make([]float32, len(arr))
	for i, a := range arr {
		switch val := a.(type) {
		case float32:
			floats[i] = val
		case float64:
			floats[i] = float32(val)
		}
	}
	return floats
}

// ReadTensorF32 reads a tensor and dequantizes to float32
func (g *GGUFFile) ReadTensorF32(name string) ([]float32, []uint64, error) {
	var info *GGUFTensorInfo
	for i := range g.Tensors {
		if g.Tensors[i].Name == name {
			info = &g.Tensors[i]
			break
		}
	}
	if info == nil {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}

	// Total number of elements
	nelems := uint64(1)
	for _, s := range info.Shape {
		nelems *= s
	}

	// Seek to tensor data
	g.file.Seek(g.DataOff+int64(info.Offset), io.SeekStart)

	out := make([]float32, nelems)

	switch info.Type {
	case GGML_TYPE_F32:
		if err := binary.Read(g.file, binary.LittleEndian, out); err != nil {
			return nil, nil, err
		}
	case GGML_TYPE_F16:
		raw := make([]uint16, nelems)
		if err := binary.Read(g.file, binary.LittleEndian, raw); err != nil {
			return nil, nil, err
		}
		for i, v := range raw {
			out[i] = float16to32(v)
		}
	case GGML_TYPE_Q4_0:
		nblocks := nelems / BLOCK_SIZE
		buf := make([]byte, nblocks*Q4_0_BYTES_PER_BLOCK)
		if _, err := io.ReadFull(g.file, buf); err != nil {
			return nil, nil, err
		}
		dequantQ4_0(buf, out, int(nblocks))
	case GGML_TYPE_Q8_0:
		nblocks := nelems / BLOCK_SIZE
		buf := make([]byte, nblocks*Q8_0_BYTES_PER_BLOCK)
		if _, err := io.ReadFull(g.file, buf); err != nil {
			return nil, nil, err
		}
		dequantQ8_0(buf, out, int(nblocks))
	default:
		return nil, nil, fmt.Errorf("unsupported tensor type: %d", info.Type)
	}

	return out, info.Shape, nil
}

func dequantQ4_0(buf []byte, out []float32, nblocks int) {
	for b := 0; b < nblocks; b++ {
		boff := b * Q4_0_BYTES_PER_BLOCK
		// Scale: first 2 bytes are float16
		scale := float16to32(uint16(buf[boff]) | uint16(buf[boff+1])<<8)

		// 16 bytes of packed 4-bit quants → 32 values
		for j := 0; j < 16; j++ {
			qbyte := buf[boff+2+j]
			lo := int(qbyte&0x0F) - 8
			hi := int(qbyte>>4) - 8

			out[b*32+j] = scale * float32(lo)
			out[b*32+16+j] = scale * float32(hi)
		}
	}
}

func dequantQ8_0(buf []byte, out []float32, nblocks int) {
	for b := 0; b < nblocks; b++ {
		boff := b * Q8_0_BYTES_PER_BLOCK
		// Scale: first 2 bytes are float16
		scale := float16to32(uint16(buf[boff]) | uint16(buf[boff+1])<<8)

		// 32 bytes of int8 quants → 32 values
		for j := 0; j < 32; j++ {
			q := int8(buf[boff+2+j])
			out[b*32+j] = scale * float32(q)
		}
	}
}

func float16to32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		// Inf/NaN
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
	}

	exp = exp + 127 - 15
	return math.Float32frombits((sign << 31) | (exp << 23) | (mant << 13))
}

func align(offset int64, alignment int64) int64 {
	return (offset + alignment - 1) & ^(alignment - 1)
}
