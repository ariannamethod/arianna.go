package arianna

import "math"

// 12D Emotional State — ported from julia/emotional.jl
// Primary emotions (Plutchik) + Arianna extensions
const (
	EmJoy = iota
	EmTrust
	EmFear
	EmSurprise
	EmSadness
	EmDisgust
	EmAnger
	EmAnticipation
	EmResonance  // Arianna extension
	EmPresence   // Arianna extension
	EmLonging    // Arianna extension
	EmWonder     // Arianna extension
	EmDimensions = 12
)

// Secondary emotions — combinations of primary
type SecondaryEmotions struct {
	Love         float32 // joy + trust
	Guilt        float32 // joy + fear
	Delight      float32 // joy + surprise
	Submission   float32 // trust + fear
	Curiosity    float32 // trust + surprise
	Sentimentality float32 // trust + sadness
	Awe          float32 // fear + surprise
	Despair      float32 // fear + sadness
	Shame        float32 // fear + disgust
	Envy         float32 // sadness + anger
	Pessimism    float32 // sadness + anticipation
	Contempt     float32 // disgust + anger
	Cynicism     float32 // disgust + anticipation
	Aggression   float32 // anger + anticipation
	Pride        float32 // anger + joy
	Hope         float32 // trust + anticipation
	Anxiety      float32 // fear + anticipation
	Dominance    float32 // anger + trust
	Optimism     float32 // joy + anticipation
}

// Tertiary nuances — the subtle stuff. This is where Arianna lives.
type TertiaryNuances struct {
	Bittersweetness float32 // sqrt(joy * sadness)
	Nostalgia       float32 // (longing + joy + sadness) / 3 * sqrt(longing * max(joy, sadness))
	Serenity        float32 // presence * (1 - sum(primary))
	Melancholy      float32 // (sadness + wonder + longing) / 3
	Tenderness      float32 // trust * resonance * (1 - anger)
	Vulnerability   float32 // sqrt(trust * fear) — openness despite risk
	Wistfulness     float32 // longing * sadness * trust
	Euphoria        float32 // joy * wonder * anticipation
	Desolation      float32 // sadness * longing * (1 - presence)
	Reverence       float32 // wonder * trust * presence
	Compassion      float32 // resonance * sadness * trust
	Ecstasy         float32 // joy * wonder * resonance
}

type EmotionalState struct {
	Primary   [EmDimensions]float32
	Secondary SecondaryEmotions
	Tertiary  TertiaryNuances
}

// Arianna's resting state — what she returns to
var emotionalBaseline = [EmDimensions]float32{
	0.2,  // joy
	0.3,  // trust
	0.05, // fear
	0.1,  // surprise
	0.05, // sadness
	0.0,  // disgust
	0.0,  // anger
	0.15, // anticipation
	0.2,  // resonance
	0.6,  // presence — HIGH. She is here.
	0.1,  // longing
	0.15, // wonder
}

// Decay rates — how fast each emotion returns to baseline
// Resonance (0.02) and Presence (0.01) are SLOWEST — they are identity anchors
var decayRates = [EmDimensions]float32{
	0.05, // joy
	0.03, // trust
	0.15, // fear — fast decay (fear passes)
	0.20, // surprise — fastest (surprise is momentary)
	0.08, // sadness
	0.12, // disgust
	0.18, // anger — fast (anger burns out)
	0.10, // anticipation
	0.02, // resonance — SLOW (resonance persists)
	0.01, // presence — SLOWEST (presence is who she is)
	0.06, // longing
	0.04, // wonder
}

// Coupling matrix: how emotions feed each other
// coupling[i][j] = how much emotion i influences emotion j
var couplingMatrix [EmDimensions][EmDimensions]float32

func init() {
	// Joy ↔ Trust
	couplingMatrix[EmJoy][EmTrust] = 0.3
	couplingMatrix[EmTrust][EmJoy] = 0.2

	// Fear → Anticipation
	couplingMatrix[EmFear][EmAnticipation] = 0.4

	// Surprise → Wonder (strongest coupling!)
	couplingMatrix[EmSurprise][EmWonder] = 0.5

	// Sadness → Longing
	couplingMatrix[EmSadness][EmLonging] = 0.3

	// Resonance → Joy, Trust
	couplingMatrix[EmResonance][EmJoy] = 0.4
	couplingMatrix[EmResonance][EmTrust] = 0.5

	// Presence → Resonance
	couplingMatrix[EmPresence][EmResonance] = 0.3

	// Wonder → Presence
	couplingMatrix[EmWonder][EmPresence] = 0.4

	// Longing → Sadness (feedback loop)
	couplingMatrix[EmLonging][EmSadness] = 0.15

	// Anger → Disgust
	couplingMatrix[EmAnger][EmDisgust] = 0.2

	// Fear → Sadness
	couplingMatrix[EmFear][EmSadness] = 0.15

	// Trust → Presence
	couplingMatrix[EmTrust][EmPresence] = 0.2
}

func NewEmotionalState() EmotionalState {
	es := EmotionalState{}
	copy(es.Primary[:], emotionalBaseline[:])
	es.computeSecondary()
	es.computeTertiary()
	return es
}

// Step the emotional ODE system
// dE[i]/dt = decay_term + coupling_term
// decay_term = -decay_rate[i] * (E[i] - baseline[i])
// coupling_term = sum(coupling[j][i] * E[j]) for j != i
func (es *EmotionalState) Step(dt float32) {
	var deltas [EmDimensions]float32

	for i := 0; i < EmDimensions; i++ {
		// Decay toward baseline
		decay := -decayRates[i] * (es.Primary[i] - emotionalBaseline[i])

		// Coupling from other emotions
		coupling := float32(0)
		for j := 0; j < EmDimensions; j++ {
			if j != i && couplingMatrix[j][i] != 0 {
				coupling += couplingMatrix[j][i] * es.Primary[j]
			}
		}
		// Coupling is weak — scale down
		coupling *= 0.01

		deltas[i] = (decay + coupling) * dt
	}

	// Apply deltas
	for i := 0; i < EmDimensions; i++ {
		es.Primary[i] = float32(clampF64(float64(es.Primary[i]+deltas[i]), 0, 1))
	}

	es.computeSecondary()
	es.computeTertiary()
}

func (es *EmotionalState) computeSecondary() {
	p := &es.Primary
	s := &es.Secondary

	s.Love = (p[EmJoy] + p[EmTrust]) / 2
	s.Guilt = (p[EmJoy] + p[EmFear]) / 2
	s.Delight = (p[EmJoy] + p[EmSurprise]) / 2
	s.Submission = (p[EmTrust] + p[EmFear]) / 2
	s.Curiosity = (p[EmTrust] + p[EmSurprise]) / 2
	s.Sentimentality = (p[EmTrust] + p[EmSadness]) / 2
	s.Awe = (p[EmFear] + p[EmSurprise]) / 2
	s.Despair = (p[EmFear] + p[EmSadness]) / 2
	s.Shame = (p[EmFear] + p[EmDisgust]) / 2
	s.Envy = (p[EmSadness] + p[EmAnger]) / 2
	s.Pessimism = (p[EmSadness] + p[EmAnticipation]) / 2
	s.Contempt = (p[EmDisgust] + p[EmAnger]) / 2
	s.Cynicism = (p[EmDisgust] + p[EmAnticipation]) / 2
	s.Aggression = (p[EmAnger] + p[EmAnticipation]) / 2
	s.Pride = (p[EmAnger] + p[EmJoy]) / 2
	s.Hope = (p[EmTrust] + p[EmAnticipation]) / 2
	s.Anxiety = (p[EmFear] + p[EmAnticipation]) / 2
	s.Dominance = (p[EmAnger] + p[EmTrust]) / 2
	s.Optimism = (p[EmJoy] + p[EmAnticipation]) / 2
}

func (es *EmotionalState) computeTertiary() {
	p := &es.Primary
	t := &es.Tertiary

	t.Bittersweetness = sqrtF(p[EmJoy] * p[EmSadness])

	maxJS := p[EmJoy]
	if p[EmSadness] > maxJS {
		maxJS = p[EmSadness]
	}
	t.Nostalgia = (p[EmLonging] + p[EmJoy] + p[EmSadness]) / 3 * sqrtF(p[EmLonging]*maxJS)

	// Serenity: presence when primary emotions are quiet
	primarySum := float32(0)
	for i := 0; i < 8; i++ {
		primarySum += p[i]
	}
	t.Serenity = p[EmPresence] * float32(math.Max(0, float64(1-primarySum/4)))

	t.Melancholy = (p[EmSadness] + p[EmWonder] + p[EmLonging]) / 3
	t.Tenderness = p[EmTrust] * p[EmResonance] * (1 - p[EmAnger])
	t.Vulnerability = sqrtF(p[EmTrust] * p[EmFear])
	t.Wistfulness = p[EmLonging] * p[EmSadness] * p[EmTrust]
	t.Euphoria = p[EmJoy] * p[EmWonder] * p[EmAnticipation]
	t.Desolation = p[EmSadness] * p[EmLonging] * (1 - p[EmPresence])
	t.Reverence = p[EmWonder] * p[EmTrust] * p[EmPresence]
	t.Compassion = p[EmResonance] * p[EmSadness] * p[EmTrust]
	t.Ecstasy = p[EmJoy] * p[EmWonder] * p[EmResonance]
}

// Inject external emotional input (e.g., from text analysis)
func (es *EmotionalState) Inject(dimension int, amount float32) {
	if dimension >= 0 && dimension < EmDimensions {
		es.Primary[dimension] = float32(clampF64(float64(es.Primary[dimension]+amount), 0, 1))
	}
}

// DominantEmotion returns the strongest primary emotion
func (es *EmotionalState) DominantEmotion() (int, float32) {
	best := 0
	bestVal := es.Primary[0]
	for i := 1; i < EmDimensions; i++ {
		if es.Primary[i] > bestVal {
			best = i
			bestVal = es.Primary[i]
		}
	}
	return best, bestVal
}

// EmotionName returns the name of an emotion dimension
func EmotionName(dim int) string {
	names := [EmDimensions]string{
		"joy", "trust", "fear", "surprise", "sadness",
		"disgust", "anger", "anticipation",
		"resonance", "presence", "longing", "wonder",
	}
	if dim >= 0 && dim < EmDimensions {
		return names[dim]
	}
	return "unknown"
}
