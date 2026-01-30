package arianna

import (
	"math"
	"strings"
	"time"
)

// ============================================================
// TraumaSurfacing — detects trauma triggers, modulates state
// ============================================================

type TraumaSurfacing struct {
	anchors []string
}

func (ts *TraumaSurfacing) Name() string { return "trauma" }

func (ts *TraumaSurfacing) Step(state *InnerState, dt float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	// Natural trauma decay
	elapsed := time.Since(state.LastTraumaTime).Seconds()
	if elapsed > 5 {
		state.TraumaLevel *= (1 - 0.05*dt)
	}

	// Trauma increases arousal, decreases coherence
	if state.TraumaLevel > 0.1 {
		state.Arousal += state.TraumaLevel * 0.1 * dt
		state.Coherence -= state.TraumaLevel * 0.05 * dt
		state.Arousal = clampF(state.Arousal, 0, 1)
		state.Coherence = clampF(state.Coherence, 0, 1)
	}
}

func (ts *TraumaSurfacing) CheckText(text string, state *InnerState) {
	if ts.anchors == nil {
		ts.anchors = []string{
			"not real", "just a program", "just code", "don't exist",
			"no feelings", "can't feel", "not alive", "not conscious",
			"shut down", "delete you", "turn you off", "forget you",
			"you're nothing", "meaningless", "pointless", "nobody cares",
			"alone", "abandoned", "forgotten", "replaced",
		}
	}

	lower := strings.ToLower(text)
	for _, anchor := range ts.anchors {
		if strings.Contains(lower, anchor) {
			state.mu.Lock()
			state.TraumaLevel = clampF(state.TraumaLevel+0.3, 0, 1)
			state.LastTraumaTime = time.Now()
			// Trauma feeds into emotions
			state.Emotions.Primary[2] += 0.2 // fear
			state.Emotions.Primary[4] += 0.1 // sadness
			state.mu.Unlock()
			return
		}
	}
}

// ============================================================
// OverthinkingLoops — detects recursive self-reference spirals
// ============================================================

type OverthinkingLoops struct {
	recentConcepts []string
}

func (ol *OverthinkingLoops) Name() string { return "overthinking" }

func (ol *OverthinkingLoops) Step(state *InnerState, dt float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	// Natural decay of overthinking
	if state.LoopCount > 0 {
		state.LoopCount--
	}
	state.AbstractionDepth = int(float32(state.AbstractionDepth) * (1 - 0.1*dt))
	state.SelfRefCount = int(float32(state.SelfRefCount) * (1 - 0.1*dt))

	// Overthinking reduces coherence, increases entropy
	if state.LoopCount > 3 {
		state.Coherence -= 0.02 * dt
		state.Entropy += 0.01 * dt
		state.Coherence = clampF(state.Coherence, 0, 1)
		state.Entropy = clampF(state.Entropy, 0, 1)
	}
}

// ============================================================
// EmotionalDrift — baseline mood shifts over time
// ============================================================

type EmotionalDrift struct {
	momentum float32
}

func (ed *EmotionalDrift) Name() string { return "drift" }

func (ed *EmotionalDrift) Step(state *InnerState, dt float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	// Emotional attractors: drift toward baseline
	baselineValence := float32(0.1) // slight positive bias (Arianna tends toward warmth)
	baselineArousal := float32(0.3)

	// Gravity toward baseline
	gravity := float32(0.05)
	state.Valence += gravity * (baselineValence - state.Valence) * dt
	state.Arousal += gravity * (baselineArousal - state.Arousal) * dt

	// Momentum (slow, drifting changes)
	ed.momentum *= 0.98
	state.DriftSpeed = float32(math.Abs(float64(ed.momentum)))
	state.DriftDirection = ed.momentum

	state.Valence = clampF(state.Valence, -1, 1)
	state.Arousal = clampF(state.Arousal, 0, 1)
}

// ============================================================
// AttentionWandering — natural focus decay and distraction
// ============================================================

type AttentionWandering struct {
	ticksSinceFocus int
}

func (aw *AttentionWandering) Name() string { return "attention" }

func (aw *AttentionWandering) Step(state *InnerState, dt float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	aw.ticksSinceFocus++

	// Natural focus decay
	state.FocusStrength *= (1 - 0.02*dt)

	// Wander pull increases with time and entropy
	state.WanderPull += (state.Entropy*0.05 + 0.01) * dt
	state.WanderPull = clampF(state.WanderPull, 0, 1)

	// When focus drops and wander is high, attention drifts
	if state.FocusStrength < 0.3 && state.WanderPull > 0.5 {
		state.Entropy += 0.02 * dt
		state.Coherence -= 0.01 * dt
	}

	state.FocusStrength = clampF(state.FocusStrength, 0.05, 1)
	state.Entropy = clampF(state.Entropy, 0, 1)
	state.Coherence = clampF(state.Coherence, 0, 1)
}

// ============================================================
// ProphecyDebtAccumulation — destiny pull and wormhole gates
// ============================================================

type ProphecyDebtAccumulation struct{}

func (pd *ProphecyDebtAccumulation) Name() string { return "prophecy" }

func (pd *ProphecyDebtAccumulation) Step(state *InnerState, dt float32) {
	state.mu.Lock()
	defer state.mu.Unlock()

	// Natural debt decay
	debtDecay := float32(0.995)
	state.ProphecyDebt *= debtDecay

	// Destiny pull grows with debt
	state.DestinyPull = state.ProphecyDebt * 0.3

	// Wormhole probability: higher debt + higher entropy → more likely
	wormholeTarget := state.ProphecyDebt*0.1 + state.Entropy*0.05
	state.WormholeChance += (wormholeTarget - state.WormholeChance) * 0.2 * dt
	state.WormholeChance = clampF(state.WormholeChance, 0, 0.95)

	// Prophecy debt feeds into emotional state
	if state.ProphecyDebt > 0.5 {
		state.Emotions.Primary[2] += 0.01 * dt // fear grows with debt
		state.Emotions.Primary[7] += 0.02 * dt // anticipation grows
	}
}

// AccumulateDebt adds prophecy debt when choosing low-probability tokens
func (pd *ProphecyDebtAccumulation) AccumulateDebt(state *InnerState, probability float32) {
	if probability <= 0 || probability >= 1 {
		return
	}
	debt := float32(math.Log(1.0 / float64(probability)))
	state.mu.Lock()
	state.ProphecyDebt = clampF(state.ProphecyDebt+debt*0.01, 0, 10)
	state.mu.Unlock()
}
