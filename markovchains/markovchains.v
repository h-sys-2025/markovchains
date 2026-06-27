module markovchains

import os
import math
import rand
import x.json2

const sep = '\x00'

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

pub fn tokenize(raw string) []string {
	punctuation := '.,!?;:\'"()-[]{}…'.split('')
	mut cleaned := raw.replace('\r\n', ' ').replace('\n', ' ').replace('\t', ' ')
	for p in punctuation {
		cleaned = cleaned.replace(p, ' ${p} ')
	}
	mut tokens := []string{}
	for part in cleaned.split(' ') {
		t := part.trim_space().to_lower()
		if t.len > 0 {
			tokens << t
		}
	}
	return tokens
}

fn join_ctx(tokens []string) string {
	mut s := ''
	for i, t in tokens {
		if i > 0 {
			s += markovchains.sep
		}
		s += t
	}
	return s
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct Config {
pub:
	order     int = 2
	smoothing f64 = 0.01
}

// ---------------------------------------------------------------------------
// Markov
// ---------------------------------------------------------------------------

pub struct Markov {
pub mut:
	model     map[string]map[string]f64
	cfg       Config
	tok_count int
}

pub fn new(cfg Config) Markov {
	return Markov{
		model: map[string]map[string]f64{}
		cfg:   cfg
	}
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

pub fn build_from_tokens(tokens []string, cfg Config) Markov {
	mut m := new(cfg)
	order := cfg.order
	if order < 1 || tokens.len <= order {
		return m
	}

	m.tok_count = tokens.len

	// Count phase
	mut counts := map[string]map[string]int{}
	for i in 0 .. tokens.len - order {
		ctx := join_ctx(tokens[i..i + order])
		nxt := tokens[i + order]
		if ctx !in counts {
			counts[ctx] = map[string]int{}
		}
		counts[ctx][nxt] = (counts[ctx][nxt] or { 0 }) + 1
	}

	// FIX: use unique vocabulary size for Laplace smoothing, not raw token count
	mut vocab_size := 0
	if cfg.smoothing > 0.0 {
		mut vocab := map[string]bool{}
		for t in tokens {
			vocab[t] = true
		}
		vocab_size = vocab.len
	}

	// Probability phase
	for ctx, next_words in counts {
		mut total := 0
		for _, c in next_words {
			total += c
		}
		smooth := cfg.smoothing
		denom := f64(total) + smooth * f64(vocab_size)

		mut probs := map[string]f64{}
		for nxt, c in next_words {
			probs[nxt] = math.round(((f64(c) + smooth) / denom) * 1_000_000.0) / 1_000_000.0
		}
		m.model[ctx] = probs.clone()
	}

	return m
}

pub fn build_from_text(text string, cfg Config) Markov {
	return build_from_tokens(tokenize(text), cfg)
}

pub fn build_from_file(path string, cfg Config) !Markov {
	raw := os.read_file(path) or { return error('cannot read file: ${err}') }
	return build_from_text(raw, cfg)
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

pub fn (m Markov) save(path string) ! {
	mut root := map[string]json2.Any{}
	root['order'] = json2.Any(m.cfg.order)
	root['smoothing'] = json2.Any(m.cfg.smoothing)
	root['tok_count'] = json2.Any(m.tok_count)

	mut transitions := map[string]json2.Any{}
	for ctx, next_map in m.model {
		mut inner := map[string]json2.Any{}
		for nxt, prob in next_map {
			inner[nxt] = json2.Any(prob)
		}
		transitions[ctx] = json2.Any(inner)
	}
	root['transitions'] = json2.Any(transitions)

	os.write_file(path, json2.encode(root, prettify: true))!
}

pub fn load(path string) !Markov {
	json_text := os.read_file(path)!
	root_any := json2.decode[json2.Any](json_text) or {
		return error('failed to decode JSON: ${err}')
	}
	root := root_any.as_map()

	cfg := Config{
		order:     int(root['order'] or { json2.Any(2) }.int())
		smoothing: root['smoothing'] or { json2.Any(0.01) }.f64()
	}
	tok_count := int(root['tok_count'] or { json2.Any(0) }.int())

	transitions_any := root['transitions'] or { json2.Any(map[string]json2.Any{}) }
	data := transitions_any.as_map()

	mut model := map[string]map[string]f64{}
	for ctx, value in data {
		inner := value.as_map()
		mut inner_map := map[string]f64{}
		for nxt, prob_any in inner {
			inner_map[nxt] = prob_any.f64()
		}
		model[ctx] = inner_map.clone()
	}

	return Markov{
		model:     model
		cfg:       cfg
		tok_count: tok_count
	}
}

// ---------------------------------------------------------------------------
// Querying
// ---------------------------------------------------------------------------

pub fn (m Markov) get_next_probabilities(ctx string) map[string]f64 {
	if ctx in m.model {
		return m.model[ctx].clone()
	}
	return map[string]f64{}
}

pub fn (m Markov) get_next_probs_for_tokens(ctx_tokens []string) map[string]f64 {
	return m.get_next_probabilities(join_ctx(ctx_tokens))
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

pub fn sample(probs map[string]f64, temperature f64) string {
	if probs.len == 0 {
		return ''
	}
	temp := if temperature <= 0.0 { 1e-8 } else { temperature }

	mut scaled := map[string]f64{}
	mut total := 0.0
	for token, prob in probs {
		p := if prob <= 0.0 { 1e-12 } else { prob }
		s := math.pow(p, 1.0 / temp)
		scaled[token] = s
		total += s
	}

	if total <= 0.0 {
		return probs.keys()[0]
	}

	r := rand.f64() * total
	mut cumulative := 0.0
	for token, s in scaled {
		cumulative += s
		if r <= cumulative {
			return token
		}
	}

	mut best := ''
	mut best_p := -1.0
	for token, prob in probs {
		if prob > best_p {
			best_p = prob
			best = token
		}
	}
	return best
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

pub struct GenerateConfig {
pub:
	max_tokens  int  = 100
	temperature f64  = 1.0
	back_off    bool = true
}

pub fn (m Markov) generate(seed []string, gcfg GenerateConfig) []string {
	order := m.cfg.order
	mut history := seed.clone()

	for history.len < order {
		history.insert(0, '')
	}

	mut output := []string{}

	for _ in 0 .. gcfg.max_tokens {
		mut picked := ''

		if gcfg.back_off {
			// FIX: filter out empty padding tokens before building context key
			for ctx_len := order; ctx_len >= 1; ctx_len-- {
				start := history.len - ctx_len
				slice := history[start..].filter(it.len > 0)
				if slice.len == 0 {
					continue
				}
				ctx := join_ctx(slice)
				probs := m.get_next_probabilities(ctx)
				if probs.len > 0 {
					picked = sample(probs, gcfg.temperature)
					break
				}
			}
		} else {
			slice := history[history.len - order..].filter(it.len > 0)
			ctx := join_ctx(slice)
			probs := m.get_next_probabilities(ctx)
			picked = sample(probs, gcfg.temperature)
		}

		if picked == '' {
			break
		}

		output << picked
		history << picked
	}

	return output
}

pub fn (m Markov) generate_text(seed_text string, gcfg GenerateConfig) string {
	seed := tokenize(seed_text)
	tokens := m.generate(seed, gcfg)

	punct_set := '.,!?;:\'"…'.split('')
	mut result := seed.join(' ')
	for tok in tokens {
		if tok in punct_set {
			result += tok
		} else {
			result += ' ' + tok
		}
	}
	return result.trim_space()
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

pub fn (m Markov) stats() string {
	return 'Markov(order=${m.cfg.order}, smoothing=${m.cfg.smoothing}, contexts=${m.model.len}, tok_count=${m.tok_count})'
}

struct Pair {
	token string
	prob  f64
}

// FIX: back-off through shorter contexts, same as generate()
pub fn (m Markov) top_continuations(ctx_tokens []string, n int) []string {
	mut probs := map[string]f64{}
	for ctx_len := ctx_tokens.len; ctx_len >= 1; ctx_len-- {
		slice := ctx_tokens[ctx_tokens.len - ctx_len..].filter(it.len > 0)
		if slice.len == 0 {
			continue
		}
		probs = m.get_next_probs_for_tokens(slice)
		if probs.len > 0 {
			break
		}
	}

	mut pairs := []Pair{}
	for tok, prob in probs {
		pairs << Pair{
			token: tok
			prob:  prob
		}
	}
	pairs.sort(a.prob > b.prob)

	mut result := []string{}
	for i, p in pairs {
		if i >= n {
			break
		}
		result << '${p.token} (${p.prob:.4f})'
	}
	return result
}