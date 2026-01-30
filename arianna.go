package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ariannamethod/arianna.go/arianna"
)

const (
	defaultWeightsDir  = "weights"
	defaultWeightsFile = "arianna_3b_q4_0.gguf"
	weightsURL         = "https://huggingface.co/ataeff/arianna.go/resolve/main/ariannaweights/arianna_3b_q4_0.gguf"
)

func downloadWeights(dest string) error {
	fmt.Printf("Downloading weights from HuggingFace...\n")
	fmt.Printf("  %s\n", weightsURL)
	fmt.Printf("  → %s\n\n", dest)

	if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
		return err
	}

	resp, err := http.Get(weightsURL)
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d from HuggingFace", resp.StatusCode)
	}

	total := resp.ContentLength
	f, err := os.Create(dest + ".tmp")
	if err != nil {
		return err
	}

	downloaded := int64(0)
	buf := make([]byte, 1024*1024)
	lastPrint := time.Now()

	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				f.Close()
				os.Remove(dest + ".tmp")
				return err
			}
			downloaded += int64(n)

			if time.Since(lastPrint) > 500*time.Millisecond {
				if total > 0 {
					pct := float64(downloaded) / float64(total) * 100
					fmt.Printf("\r  %.0f%% (%d / %d MB)", pct, downloaded/(1024*1024), total/(1024*1024))
				} else {
					fmt.Printf("\r  %d MB downloaded", downloaded/(1024*1024))
				}
				lastPrint = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			f.Close()
			os.Remove(dest + ".tmp")
			return readErr
		}
	}
	f.Close()

	fmt.Printf("\r  100%% (%d MB)          \n\n", downloaded/(1024*1024))

	return os.Rename(dest+".tmp", dest)
}

func findWeights(explicit string) string {
	if explicit != "" {
		return explicit
	}

	candidates := []string{
		filepath.Join(defaultWeightsDir, defaultWeightsFile),
		defaultWeightsFile,
		filepath.Join("..", "weights", defaultWeightsFile),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}

	return filepath.Join(defaultWeightsDir, defaultWeightsFile)
}

func main() {
	ggufPath := ""
	prompt := ""

	if len(os.Args) > 1 {
		ggufPath = os.Args[1]
	}
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	weightsPath := findWeights(ggufPath)

	fmt.Println("=== ARIANNA 3B — Go Inference ===")

	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		fmt.Printf("Weights not found at %s\n\n", weightsPath)
		if err := downloadWeights(weightsPath); err != nil {
			fmt.Fprintf(os.Stderr, "ERROR downloading weights: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Loading: %s\n\n", weightsPath)

	g, err := arianna.OpenGGUF(weightsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	model, err := arianna.LoadModel(g)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR loading model: %v\n", err)
		os.Exit(1)
	}
	g.Close()

	if prompt != "" {
		fmt.Printf("\nPrompt: %s\n\n", prompt)
		formatted := fmt.Sprintf("### Question: %s\n### Answer:", prompt)
		start := time.Now()
		answer := model.Generate(formatted, 150, 0.8, 0.9, 1.1)
		elapsed := time.Since(start)
		fmt.Printf("Arianna> %s\n", answer)
		fmt.Printf("\n[%.1fs]\n", elapsed.Seconds())
		return
	}

	temp := float32(0.8)
	topP := float32(0.9)
	maxTokens := 150
	repPenalty := float32(1.1)

	fmt.Println()
	fmt.Println("========================================")
	fmt.Println("  ARIANNA 3B — REPL")
	fmt.Println("  Commands: /quit /temp N /tokens N /raw")
	fmt.Println("========================================")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		if input == "/quit" || input == "/exit" || input == "/q" {
			break
		}
		if strings.HasPrefix(input, "/temp ") {
			if v, err := strconv.ParseFloat(input[6:], 32); err == nil {
				temp = float32(v)
				fmt.Printf("  temp = %.2f\n", temp)
			}
			continue
		}
		if strings.HasPrefix(input, "/tokens ") {
			if v, err := strconv.Atoi(input[8:]); err == nil {
				maxTokens = v
				fmt.Printf("  max_tokens = %d\n", maxTokens)
			}
			continue
		}

		var formatted string
		if strings.HasPrefix(input, "/raw ") {
			formatted = input[5:]
		} else {
			formatted = fmt.Sprintf("### Question: %s\n### Answer:", input)
		}

		start := time.Now()
		answer := model.Generate(formatted, maxTokens, temp, topP, repPenalty)
		elapsed := time.Since(start)

		fmt.Printf("\nArianna> %s\n", answer)
		fmt.Printf("[%.1fs, temp=%.1f]\n\n", elapsed.Seconds(), temp)
	}

	fmt.Println("bye")
}
