package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	ggufPath := "../weights/arianna_3b_q4_0.gguf"
	if len(os.Args) > 1 {
		ggufPath = os.Args[1]
	}

	fmt.Println("=== ARIANNA 3B — Go Inference ===")
	fmt.Printf("Loading: %s\n\n", ggufPath)

	// Open GGUF
	g, err := OpenGGUF(ggufPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	// Load model (weights + tokenizer)
	model, err := LoadModel(g)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR loading model: %v\n", err)
		os.Exit(1)
	}
	g.Close()

	// Single prompt mode
	if len(os.Args) > 2 {
		prompt := os.Args[2]
		fmt.Printf("\nPrompt: %s\n\n", prompt)
		formatted := fmt.Sprintf("### Question: %s\n### Answer:", prompt)
		start := time.Now()
		answer := model.Generate(formatted, 150, 0.8, 0.9, 1.1)
		elapsed := time.Since(start)
		fmt.Printf("Arianna> %s\n", answer)
		fmt.Printf("\n[%.1fs]\n", elapsed.Seconds())
		return
	}

	// REPL mode
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

		// Commands
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

		// /raw — send raw prompt without formatting
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
