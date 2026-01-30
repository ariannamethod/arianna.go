package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// SentencePiece BPE tokenizer loaded from GGUF metadata
type Tokenizer struct {
	Tokens    []string  // id → token string (e.g. "▁hello")
	Scores    []float32 // id → score (merge priority)
	VocabSize int

	// Special tokens
	BOS int // beginning of sequence
	EOS int // end of sequence
	UNK int // unknown

	// Reverse lookup: token string → id
	tokenToID map[string]int
}

// LoadTokenizerFromGGUF extracts tokenizer from GGUF metadata
func LoadTokenizerFromGGUF(g *GGUFFile) (*Tokenizer, error) {
	tokens := g.GetStringArray("tokenizer.ggml.tokens")
	if tokens == nil {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF")
	}

	scores := g.GetFloat32Array("tokenizer.ggml.scores")

	t := &Tokenizer{
		Tokens:    tokens,
		Scores:    scores,
		VocabSize: len(tokens),
		BOS:       1,  // <s>
		EOS:       2,  // </s>
		UNK:       0,  // <unk>
		tokenToID: make(map[string]int, len(tokens)),
	}

	// Build reverse lookup
	for i, tok := range tokens {
		t.tokenToID[tok] = i
	}

	// Try to get special token IDs from metadata
	if v := g.GetUint32("tokenizer.ggml.bos_token_id"); v > 0 {
		t.BOS = int(v)
	}
	if v := g.GetUint32("tokenizer.ggml.eos_token_id"); v > 0 {
		t.EOS = int(v)
	}
	if v := g.GetUint32("tokenizer.ggml.unknown_token_id"); v > 0 {
		t.UNK = int(v)
	}

	fmt.Printf("Tokenizer: %d tokens, BOS=%d, EOS=%d, UNK=%d\n",
		t.VocabSize, t.BOS, t.EOS, t.UNK)

	return t, nil
}

// Encode text to token IDs using SentencePiece BPE
func (t *Tokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	// SentencePiece: add ▁ (U+2581) at the start to represent leading space
	// and replace all spaces with ▁
	processed := strings.ReplaceAll(text, " ", "▁")
	if !strings.HasPrefix(processed, "▁") {
		processed = "▁" + processed
	}

	// Start with each UTF-8 character as a separate token
	// Try to find each char in vocab, fall back to byte tokens <0xNN>
	symbols := make([]string, 0)
	for _, r := range processed {
		symbols = append(symbols, string(r))
	}

	// Convert initial characters to token IDs where possible
	// For unknown chars, use byte fallback tokens <0xNN>
	tokens := make([]string, 0, len(symbols))
	for _, s := range symbols {
		if _, ok := t.tokenToID[s]; ok {
			tokens = append(tokens, s)
		} else {
			// Byte fallback: encode each byte as <0xNN>
			for _, b := range []byte(s) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				tokens = append(tokens, byteToken)
			}
		}
	}

	// BPE merge loop: repeatedly merge the highest-scoring pair
	for {
		if len(tokens) < 2 {
			break
		}

		// Find the best pair to merge (highest score)
		bestScore := float32(-math.MaxFloat32)
		bestIdx := -1

		for i := 0; i < len(tokens)-1; i++ {
			merged := tokens[i] + tokens[i+1]
			id, ok := t.tokenToID[merged]
			if ok && t.Scores[id] > bestScore {
				bestScore = t.Scores[id]
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // No more merges possible
		}

		// Merge the best pair
		merged := tokens[bestIdx] + tokens[bestIdx+1]
		newTokens := make([]string, 0, len(tokens)-1)
		newTokens = append(newTokens, tokens[:bestIdx]...)
		newTokens = append(newTokens, merged)
		newTokens = append(newTokens, tokens[bestIdx+2:]...)
		tokens = newTokens
	}

	// Convert token strings to IDs
	ids := make([]int, 1, len(tokens)+1)
	ids[0] = t.BOS // Prepend BOS — HuggingFace adds it during training
	for _, tok := range tokens {
		id, ok := t.tokenToID[tok]
		if ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.UNK)
		}
	}

	return ids
}

// Decode token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= t.VocabSize {
			continue
		}
		// Skip special tokens
		if id == t.BOS || id == t.EOS || id == t.UNK {
			continue
		}
		tok := t.Tokens[id]
		// Skip byte tokens that look like <0xNN> for now
		if strings.HasPrefix(tok, "<0x") && strings.HasSuffix(tok, ">") {
			// Decode byte token
			var b byte
			fmt.Sscanf(tok, "<0x%02X>", &b)
			sb.WriteByte(b)
			continue
		}
		sb.WriteString(tok)
	}

	// SentencePiece: ▁ → space
	result := sb.String()
	result = strings.ReplaceAll(result, "▁", " ")
	// Trim leading space (artifact of SentencePiece)
	if strings.HasPrefix(result, " ") {
		result = result[1:]
	}
	return result
}

// DecodeOne decodes a single token ID to string
func (t *Tokenizer) DecodeOne(id int) string {
	if id < 0 || id >= t.VocabSize {
		return ""
	}
	if id == t.BOS || id == t.EOS || id == t.UNK {
		return ""
	}
	tok := t.Tokens[id]
	if strings.HasPrefix(tok, "<0x") && strings.HasSuffix(tok, ">") {
		var b byte
		fmt.Sscanf(tok, "<0x%02X>", &b)
		return string(b)
	}
	return strings.ReplaceAll(tok, "▁", " ")
}

// TopK returns the top-k token IDs by logit value
func TopK(logits []float32, k int) []int {
	type scored struct {
		id    int
		score float32
	}
	items := make([]scored, len(logits))
	for i, v := range logits {
		items[i] = scored{i, v}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})
	if k > len(items) {
		k = len(items)
	}
	result := make([]int, k)
	for i := 0; i < k; i++ {
		result[i] = items[i].id
	}
	return result
}
