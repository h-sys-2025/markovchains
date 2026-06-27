// markovchains — a Markov chain library for *anything*
//
// Works on words, characters, bytes, DNA bases, music notes, emoji,
// log lines, or any []string you feed it.
//
// Quick-start:
//   m := markov.from_text("your corpus here", markov.cfg())
//   println(m.next("hello"))           // single next token
//   println(m.complete("hello", 20))   // full generated string
//
module markovchains

import os
import math
import rand
//import x.json2

// sep is the internal key separator.  Pipe is human-readable in JSON
// and stripped from text input by the tokenizer so it never collides.
const sep = "|"

// =============================================================================
// Tokenizers — convert raw input into []string tokens
// =============================================================================

// words  — default text tokenizer.
// Lowercases, keeps punctuation as separate tokens.
pub fn words(raw string) []string {
	punct := ".,!?;:\"()-[]{}…|/\\@#$%^&*+=<>~`".split("")
	mut s := raw.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
	for p in punct {
		s = s.replace(p, " ${p} ")
	}
	mut out := []string{}
	for part in s.split(" ") {
		t := part.trim_space().to_lower()
		if t.len > 0 {
			out << t
		}
	}
	return out
}

// chars — character-level tokenizer.
// Each Unicode grapheme becomes one token. Good for name/word generation.
pub fn chars(raw string) []string {
	mut out := []string{}
	for ch in raw.runes() {
		s := ch.str()
		if s != " " && s != "\n" && s != "\r" && s != "\t" {
			out << s.to_lower()
		}
	}
	return out
}

// lines — one token per non-empty line.
// Good for modelling sequences of log lines, sentences, moves, etc.
pub fn lines(raw string) []string {
	mut out := []string{}
	for line in raw.split("\n") {
		t := line.trim_space()
		if t.len > 0 {
			out << t
		}
	}
	return out
}

// split_by — tokenize by any custom delimiter.
// e.g. split_by("C,G,A,T,G", ",")  for DNA
//      split_by("60 62 64 65", " ") for MIDI notes
pub fn split_by(raw string, delimiter string) []string {
	mut out := []string{}
	for part in raw.split(delimiter) {
		t := part.trim_space()
		if t.len > 0 {
			out << t
		}
	}
	return out
}

// =============================================================================
// Config
// =============================================================================

pub struct Config {
pub:
	// order: how many tokens of context to use.
	//   1 → bigram:   "A"     → "B"
	//   2 → trigram:  "A|B"   → "C"   (default, good balance)
	//   3 → 4-gram:   "A|B|C" → "D"   (needs large corpus)
	order int = 2

	// smoothing: Laplace additive smoothing.
	//   0.0  → off (exact corpus counts, can get stuck)
	//   0.01 → light (recommended)
	//   0.1+ → heavy (very random output)
	smoothing f64 = 0.01
}

// cfg returns the default Config. Sugar for the common case.
pub fn cfg() Config {
	return Config{}
}

// cfg_order returns a Config with a custom order and default smoothing.
pub fn cfg_order(order int) Config {
	return Config{ order: order }
}

// =============================================================================
// Core struct
// =============================================================================

pub struct Markov {
pub mut:
	// model[context_key] = map of next_token → probability
	model     map[string]map[string]f64
	cfg       Config
	tok_count int   // total tokens seen during training
	vocab     int   // unique token count
}

// =============================================================================
// Builders
// =============================================================================

// from_tokens trains on a pre-tokenized []string.
// This is the core builder — everything else calls this.
pub fn from_tokens(tokens []string, c Config) Markov {
	mut m := Markov{ cfg: c }
	order := c.order
	if order < 1 || tokens.len <= order {
		return m
	}
	m.tok_count = tokens.len

	// ── count phase ──────────────────────────────────────────────────────────
	mut counts := map[string]map[string]int{}
	for i in 0 .. tokens.len - order {
		ctx := join_ctx(tokens[i..i + order])
		nxt := tokens[i + order]
		if ctx !in counts {
			counts[ctx] = map[string]int{}
		}
		counts[ctx][nxt] = (counts[ctx][nxt] or { 0 }) + 1
	}

	// ── vocabulary (unique tokens) for Laplace denominator ───────────────────
	mut vocab_map := map[string]bool{}
	for t in tokens { vocab_map[t] = true }
	m.vocab = vocab_map.len
	vocab_sz := if c.smoothing > 0.0 { m.vocab } else { 0 }

	// ── probability phase ─────────────────────────────────────────────────────
	for ctx, nxt_counts in counts {
		mut total := 0
		for _, cnt in nxt_counts { total += cnt }
		denom := f64(total) + c.smoothing * f64(vocab_sz)

		mut probs := map[string]f64{}
		for nxt, cnt in nxt_counts {
			raw := (f64(cnt) + c.smoothing) / denom
			probs[nxt] = math.round(raw * 1_000_000.0) / 1_000_000.0
		}
		m.model[ctx] = probs.clone()
	}
	return m
}

// from_text tokenizes with the words() tokenizer then trains.
pub fn from_text(text string, c Config) Markov {
	return from_tokens(words(text), c)
}

// from_chars trains a character-level model.
pub fn from_chars(text string, c Config) Markov {
	return from_tokens(chars(text), c)
}

// from_lines trains one-token-per-line.
pub fn from_lines(text string, c Config) Markov {
	return from_tokens(lines(text), c)
}

// from_file reads a corpus file and trains with the words() tokenizer.
pub fn from_file(path string, c Config) !Markov {
	raw := os.read_file(path) or { return error("cannot read '${path}': ${err}") }
	return from_text(raw, c)
}

// from_file_chars reads a corpus file and trains a char-level model.
pub fn from_file_chars(path string, c Config) !Markov {
	raw := os.read_file(path) or { return error("cannot read '${path}': ${err}") }
	return from_chars(raw, c)
}

// from_files trains on multiple corpus files merged together.
pub fn from_files(paths []string, c Config) !Markov {
	mut all_tokens := []string{}
	for path in paths {
		raw := os.read_file(path) or { return error("cannot read '${path}': ${err}") }
		all_tokens << words(raw)
	}
	return from_tokens(all_tokens, c)
}

// =============================================================================
// Merging
// =============================================================================

// merge combines two models trained with the same Config.
// Probabilities are re-weighted by their token counts so larger corpora
// contribute proportionally more.
pub fn (a Markov) merge(b Markov) Markov {
	w_a := f64(a.tok_count)
	w_b := f64(b.tok_count)
	total_w := w_a + w_b
	if total_w == 0 { return a }

	mut m := Markov{
		cfg:       a.cfg
		tok_count: a.tok_count + b.tok_count
		vocab:     a.vocab + b.vocab   // approximate
	}

	// Collect all context keys from both models
	mut all_ctx := map[string]bool{}
	for ctx, _ in a.model { all_ctx[ctx] = true }
	for ctx, _ in b.model { all_ctx[ctx] = true }

	for ctx, _ in all_ctx {
		mut probs := map[string]f64{}

		// Weighted blend of both distributions
		if ctx in a.model {
			for tok, p in a.model[ctx] {
				probs[tok] = (probs[tok] or { 0.0 }) + p * (w_a / total_w)
			}
		}
		if ctx in b.model {
			for tok, p in b.model[ctx] {
				probs[tok] = (probs[tok] or { 0.0 }) + p * (w_b / total_w)
			}
		}
		m.model[ctx] = probs.clone()
	}
	return m
}

// train_more adds more text to an already-trained model by merging.
pub fn (m Markov) train_more(text string) Markov {
	extra := from_text(text, m.cfg)
	return m.merge(extra)
}

// train_more_file adds a new corpus file to an existing model.
pub fn (m Markov) train_more_file(path string) !Markov {
	raw := os.read_file(path) or { return error("cannot read '${path}': ${err}") }
	return m.train_more(raw)
}

// =============================================================================
// Internal helpers
// =============================================================================

fn join_ctx(tokens []string) string {
	mut s := ""
	for i, t in tokens {
		if i > 0 { s += markovchains.sep }
		s += t
	}
	return s
}

// probs_for returns the probability distribution for a context slice,
// with automatic back-off to shorter contexts when there"s no match.
fn (m Markov) probs_for(ctx_tokens []string) map[string]f64 {
    if ctx_tokens.len == 0 {
        return map[string]f64{}
    }
    for ctx_len := ctx_tokens.len; ctx_len >= 1; ctx_len-- {
        slice := ctx_tokens[ctx_tokens.len - ctx_len..].filter(it.len > 0 && it != "<start>")
        if slice.len == 0 {
            continue
        }
        key := join_ctx(slice)
        if key in m.model {
            return m.model[key].clone()
        }
    }
    return map[string]f64{}
}

// =============================================================================
// Sampling
// =============================================================================

// sample_from picks one token from a probability map using temperature scaling.
//   temperature 1.0 → unchanged  |  < 1.0 → sharper  |  > 1.0 → more random
pub fn sample_from(probs map[string]f64, temperature f64) string {
	if probs.len == 0 { return "" }
	temp := if temperature <= 0.0 { 1e-9 } else { temperature }

	mut scaled := map[string]f64{}
	mut total := 0.0
	for tok, prob in probs {
		p := if prob <= 0.0 { 1e-12 } else { prob }
		s := math.pow(p, 1.0 / temp)
		scaled[tok] = s
		total += s
	}
	if total <= 0.0 { return probs.keys()[0] }

	r := rand.f64() * total
	mut cumulative := 0.0
	for tok, s in scaled {
		cumulative += s
		if r <= cumulative { return tok }
	}

	// fallback: argmax
	mut best := ""
	mut best_p := -1.0
	for tok, prob in probs {
		if prob > best_p { best_p = prob; best = tok }
	}
	return best
}

// top_k filters a probability map to the k highest-probability tokens,
// then renormalises. Use before sample_from for more focused output.
pub fn top_k(probs map[string]f64, k int) map[string]f64 {
	if probs.len == 0 || k <= 0 { return probs }
	mut pairs := []Pair{}
	for tok, prob in probs { pairs << Pair{ tok, prob } }
	pairs.sort(a.prob > b.prob)

	mut out := map[string]f64{}
	mut total := 0.0
	for i, p in pairs {
		if i >= k { break }
		out[p.token] = p.prob
		total += p.prob
	}
	// renormalise
	if total > 0 {
		for tok, _ in out { out[tok] = out[tok] / total }
	}
	return out
}

// top_p filters to the smallest set of tokens whose cumulative probability
// exceeds threshold p (nucleus sampling). Great for creative text.
pub fn top_p(probs map[string]f64, p f64) map[string]f64 {
	if probs.len == 0 { return probs }
	mut pairs := []Pair{}
	for tok, prob in probs { pairs << Pair{ tok, prob } }
	pairs.sort(a.prob > b.prob)

	mut out := map[string]f64{}
	mut cumulative := 0.0
	for pair in pairs {
		out[pair.token] = pair.prob
		cumulative += pair.prob
		if cumulative >= p { break }
	}
	// renormalise
	mut total := 0.0
	for _, prob in out { total += prob }
	if total > 0 {
		for tok, _ in out { out[tok] = out[tok] / total }
	}
	return out
}

// =============================================================================
// Simple one-liner API  (the "easy as fuck" part)
// =============================================================================

// next returns the single most likely next token for a seed string.
// Uses temperature 1.0 and full back-off.
pub fn (m Markov) next(seed string) string {
	ctx := words(seed)
	probs := m.probs_for(ctx)
	return sample_from(probs, 1.0)
}

// next_n returns n independently sampled next tokens (not a chain).
pub fn (m Markov) next_n(seed string, n int) []string {
    ctx := words(seed)
    probs := m.probs_for(ctx)
    mut out := []string{}
    for _ in 0 .. n {
        if probs.len == 0 {
            out << ""
        } else {
            out << sample_from(probs, 1.0)
        }
    }
    return out
}
// complete generates up to max_tokens tokens after seed and returns the
// full string (seed + generated), with punctuation re-attached.
pub fn (m Markov) complete(seed string, max_tokens int) string {
	gcfg := GenConfig{ max_tokens: max_tokens }
	return m.generate_text(seed, gcfg)
}

// walk returns a raw []string token chain — useful for non-text data.
pub fn (m Markov) walk(seed []string, steps int) []string {
	gcfg := GenConfig{ max_tokens: steps }
	return m.generate(seed, gcfg)
}

// =============================================================================
// Full generation API
// =============================================================================

pub struct GenConfig {
pub:
    max_tokens  int   = 100
    temperature f64   = 1.0
    back_off    bool  = true
    top_k_n     int   = 0
    top_p_val   f64   = 0.0
    stop_token  string = ""
}

// generate returns a []string of generated tokens starting from seed tokens.
pub fn (m Markov) generate(seed []string, gcfg GenConfig) []string {
    order := m.cfg.order
    mut history := seed.clone()

    // Better padding: only pad if needed, and use a special start token or truncate
    for history.len < order {
        history.insert(0, "<start>")
    }

    mut output := []string{}

    for _ in 0 .. gcfg.max_tokens {
        mut ctx_slice := history[history.len - order..].clone()

        // Remove padding tokens for lookup
        ctx_slice = ctx_slice.filter(it != "<start>" && it.len > 0)

        mut probs := map[string]f64{}
        if gcfg.back_off {
            probs = m.probs_for(ctx_slice)
        } else if ctx_slice.len == order {
            key := join_ctx(ctx_slice)
            if key in m.model {
                probs = m.model[key].clone()
            }
        }

        if probs.len == 0 {
            // Try shorter context
            if ctx_slice.len > 1 {
                probs = m.probs_for(ctx_slice[1..])
            }
            if probs.len == 0 {
                break
            }
        }

        mut filtered := probs.clone()
        if gcfg.top_k_n > 0 {
            filtered = top_k(filtered, gcfg.top_k_n)
        }
        if gcfg.top_p_val > 0 {
            filtered = top_p(filtered, gcfg.top_p_val)
        }

        picked := sample_from(filtered, gcfg.temperature)
        if picked == "" || picked == "<start>" {
            break
        }

        output << picked
        history << picked

        if gcfg.stop_token != "" && picked == gcfg.stop_token {
            break
        }
    }

    return output
}

// generate_text tokenizes seed_text, generates, then re-joins into a string.
pub fn (m Markov) generate_text(seed_text string, gcfg GenConfig) string {
	seed := words(seed_text)
	tokens := m.generate(seed, gcfg)
	return pretty_join(seed, tokens)
}

// generate_chars generates character-level output and joins without spaces.
pub fn (m Markov) generate_chars(seed_text string, gcfg GenConfig) string {
	seed := chars(seed_text)
	tokens := m.generate(seed, gcfg)
	return seed.join("") + tokens.join("")
}

// generate_seq generates from arbitrary token sequences (DNA, music, etc.).
pub fn (m Markov) generate_seq(seed []string, gcfg GenConfig) []string {
	return m.generate(seed, gcfg)
}

// pretty_join re-attaches punctuation and capitalises the first letter.
fn pretty_join(seed []string, generated []string) string {
	punct_set := ".,!?;:\"…".split("")
	mut result := seed.join(" ")
	for tok in generated {
		if tok in punct_set {
			result += tok
		} else {
			result += " " + tok
		}
	}
	return result.trim_space()
}

// =============================================================================
// Querying / inspection
// =============================================================================

// prob returns the probability of `next_token` following the given context.
// Returns 0.0 if unseen.
pub fn (m Markov) prob(ctx_tokens []string, next_token string) f64 {
	probs := m.probs_for(ctx_tokens)
	return probs[next_token] or { 0.0 }
}

// top returns the top-n most likely next tokens and their probabilities
// for a given context (with back-off).
pub fn (m Markov) top(ctx_tokens []string, n int) []Prediction {
	probs := m.probs_for(ctx_tokens)
	mut pairs := []Pair{}
	for tok, prob in probs { pairs << Pair{ tok, prob } }
	pairs.sort(a.prob > b.prob)

	mut out := []Prediction{}
	for i, p in pairs {
		if i >= n { break }
		out << Prediction{ token: p.token, prob: p.prob }
	}
	return out
}

// knows returns true if the model has seen this context.
pub fn (m Markov) knows(ctx_tokens []string) bool {
	return m.probs_for(ctx_tokens).len > 0
}

// random_start returns a random starting context from the model.
// Useful when you don"t have a seed.
pub fn (m Markov) random_start() []string {
    keys := m.model.keys()
    if keys.len == 0 {
        return []string{}
    }
    key := keys[rand.intn(keys.len) or { 0 }]
    mut parts := key.split(markovchains.sep)
    // Filter out any empty parts
    return parts.filter(it.len > 0)
}

// =============================================================================
// Prediction struct
// =============================================================================

pub struct Prediction {
pub:
	token string
	prob  f64
}

pub fn (p Prediction) str() string {
	return "\"${p.token}\" (${p.prob:.4f})"
}

struct Pair {
	token string
	prob  f64
}

// =============================================================================
// Stats
// =============================================================================

pub fn (m Markov) stats() string {
	return "Markov(order=${m.cfg.order}, smoothing=${m.cfg.smoothing}, " +
		"contexts=${m.model.len}, vocab=${m.vocab}, tok_count=${m.tok_count})"
}

// perplexity estimates model quality on a test string (lower = better fit).
// Returns 0 if the string can"t be evaluated.
pub fn (m Markov) perplexity(test_text string) f64 {
	toks := words(test_text)
	order := m.cfg.order
	if toks.len <= order { return 0.0 }

	mut log_sum := 0.0
	mut count := 0
	for i in 0 .. toks.len - order {
		ctx := toks[i..i + order]
		nxt := toks[i + order]
		p := m.prob(ctx, nxt)
		if p > 0.0 {
			log_sum += math.log2(p)
			count++
		}
	}
	if count == 0 { return 0.0 }
	return math.pow(2.0, -log_sum / f64(count))
}

// =============================================================================
// Save / Load
// =============================================================================

pub fn (m Markov) save(path string) ! {
	mut root := map[string]json2.Any{}
	root["order"]     = json2.Any(m.cfg.order)
	root["smoothing"] = json2.Any(m.cfg.smoothing)
	root["tok_count"] = json2.Any(m.tok_count)
	root["vocab"]     = json2.Any(m.vocab)

	mut transitions := map[string]json2.Any{}
	for ctx, next_map in m.model {
		mut inner := map[string]json2.Any{}
		for nxt, prob in next_map { inner[nxt] = json2.Any(prob) }
		transitions[ctx] = json2.Any(inner)
	}
	root["transitions"] = json2.Any(transitions)

	os.write_file(path, json2.encode(root, prettify: true))!
}

pub fn load(path string) !Markov {
	json_text := os.read_file(path)!
	root_any  := json2.decode[json2.Any](json_text) or {
		return error("failed to decode JSON: ${err}")
	}
	root := root_any.as_map()

	c := Config{
		order:     int(root["order"] or { json2.Any(2) }.int())
		smoothing: root["smoothing"] or { json2.Any(0.01) }.f64()
	}

	mut model := map[string]map[string]f64{}
	transitions_any := root["transitions"] or { json2.Any(map[string]json2.Any{}) }
	for ctx, val in transitions_any.as_map() {
		mut inner := map[string]f64{}
		for nxt, prob_any in val.as_map() { inner[nxt] = prob_any.f64() }
		model[ctx] = inner.clone()
	}

	return Markov{
		model:     model
		cfg:       c
		tok_count: int(root["tok_count"] or { json2.Any(0) }.int())
		vocab:     int(root["vocab"] or { json2.Any(0) }.int())
	}
}