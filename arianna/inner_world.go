package arianna

import (
	"math"
	"sync"
	"time"
)

// Signal types for inter-process communication
type SignalType int

const (
	SignalTrauma SignalType = iota
	SignalOverthink
	SignalDrift
	SignalMemory
	SignalAttention
	SignalProphecy
	SignalWarmth
	SignalCoherence
)

type Signal struct {
	Type      SignalType
	Value     float32
	Source    string
	Timestamp time.Time
}

// InnerState — the shared state of Arianna's inner world
// Ported from arianna.c/inner_world/types.go
type InnerState struct {
	mu sync.RWMutex

	// Emotional baseline
	Arousal   float32
	Valence   float32
	Entropy   float32
	Coherence float32

	// Trauma
	TraumaLevel    float32
	LastTraumaTime time.Time

	// Overthinking
	LoopCount       int
	AbstractionDepth int
	SelfRefCount    int

	// Emotional drift
	DriftDirection float32
	DriftSpeed     float32

	// Attention
	FocusStrength float32
	WanderPull    float32

	// Prophecy physics
	ProphecyDebt   float32
	DestinyPull    float32
	WormholeChance float32

	// 12D Emotional state (from Julia emotional.jl)
	Emotions EmotionalState

	// Generation modulation (computed from inner state)
	TempMod    float32 // temperature multiplier
	TopPMod    float32 // top_p multiplier
	RepMod     float32 // repetition penalty multiplier
}

func NewInnerState() *InnerState {
	s := &InnerState{
		Arousal:        0.3,
		Valence:        0.1,
		Coherence:      0.7,
		FocusStrength:  0.8,
		TempMod:        1.0,
		TopPMod:        1.0,
		RepMod:         1.0,
	}
	s.Emotions = NewEmotionalState()
	return s
}

// Process interface — every inner process implements this
type Process interface {
	Name() string
	Step(state *InnerState, dt float32)
}

// InnerWorld — orchestrator for all inner processes
type InnerWorld struct {
	State     *InnerState
	Signals   chan Signal
	processes []Process
	stopChan  chan struct{}
	running   bool
	mu        sync.Mutex
}

func NewInnerWorld() *InnerWorld {
	iw := &InnerWorld{
		State:    NewInnerState(),
		Signals:  make(chan Signal, 256),
		stopChan: make(chan struct{}),
	}

	// Register all inner processes
	iw.processes = []Process{
		&TraumaSurfacing{},
		&OverthinkingLoops{},
		&EmotionalDrift{},
		&AttentionWandering{},
		&ProphecyDebtAccumulation{},
	}

	return iw
}

// Start background goroutines
func (iw *InnerWorld) Start() {
	iw.mu.Lock()
	defer iw.mu.Unlock()
	if iw.running {
		return
	}
	iw.running = true

	// Signal router
	go iw.routeSignals()
}

// Stop all goroutines
func (iw *InnerWorld) Stop() {
	iw.mu.Lock()
	defer iw.mu.Unlock()
	if !iw.running {
		return
	}
	iw.running = false
	close(iw.stopChan)
}

// Step all processes synchronously (called per token)
func (iw *InnerWorld) Step(dt float32) {
	for _, proc := range iw.processes {
		proc.Step(iw.State, dt)
	}

	// Step emotional ODE
	iw.State.mu.Lock()
	iw.State.Emotions.Step(dt)
	iw.computeModulation()
	iw.State.mu.Unlock()
}

// Compute generation parameter modulations from inner state
func (iw *InnerWorld) computeModulation() {
	s := iw.State

	// Temperature: arousal increases temp, trauma decreases it
	tempMod := float32(1.0)
	tempMod += (s.Arousal - 0.3) * 0.5  // arousal above baseline → warmer
	tempMod -= s.TraumaLevel * 0.3       // trauma → more conservative
	tempMod += s.Emotions.Tertiary.Euphoria * 0.2
	tempMod -= s.Emotions.Tertiary.Desolation * 0.2
	s.TempMod = clampF(tempMod, 0.5, 1.5)

	// Top-p: coherence narrows, entropy widens
	topPMod := float32(1.0)
	topPMod -= (s.Coherence - 0.5) * 0.2 // high coherence → narrower
	topPMod += s.Entropy * 0.1            // high entropy → wider
	s.TopPMod = clampF(topPMod, 0.7, 1.3)

	// Repetition penalty: overthinking increases it
	repMod := float32(1.0)
	repMod += float32(s.LoopCount) * 0.05 // more loops → penalize repeats harder
	s.RepMod = clampF(repMod, 1.0, 1.5)
}

// ProcessText feeds text to inner world for analysis
func (iw *InnerWorld) ProcessText(text string) {
	// Check for trauma triggers
	if ts, ok := iw.findProcess("trauma").(*TraumaSurfacing); ok {
		ts.CheckText(text, iw.State)
	}

	// Update attention
	iw.State.mu.Lock()
	iw.State.FocusStrength = clampF(iw.State.FocusStrength+0.1, 0, 1)
	iw.State.mu.Unlock()
}

func (iw *InnerWorld) findProcess(name string) Process {
	for _, p := range iw.processes {
		if p.Name() == name {
			return p
		}
	}
	return nil
}

// Emit signal (non-blocking)
func (iw *InnerWorld) emit(sig Signal) {
	select {
	case iw.Signals <- sig:
	default:
	}
}

func (iw *InnerWorld) routeSignals() {
	for {
		select {
		case <-iw.stopChan:
			return
		case sig := <-iw.Signals:
			iw.State.mu.Lock()
			switch sig.Type {
			case SignalTrauma:
				iw.State.TraumaLevel = clampF(iw.State.TraumaLevel+sig.Value, 0, 1)
			case SignalCoherence:
				iw.State.Coherence = clampF(sig.Value, 0, 1)
			case SignalWarmth:
				iw.State.Emotions.Primary[0] += sig.Value * 0.1 // joy
			}
			iw.State.mu.Unlock()
		}
	}
}

func clampF(v, lo, hi float32) float32 {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

func clampF64(v, lo, hi float64) float64 {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

func sqrtF(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
