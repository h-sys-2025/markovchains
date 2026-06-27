module markovchains

import os
import math
import rand
import x.json2

// SEP is the internal separator used to join multi-token context keys.
// Using a non-printable byte means it can never collide with real text.
const sep = '\x00'

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

// tokenize splits raw text into lowercase tokens, keeping punctuation as
// separate tokens so the model can learn punctuation placement.
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

// join_ctx joins a slice of tokens into a single context key.
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
  // order is the n-gram context length.
  // 1 = bigram  ("A" → "B")
  // 2 = trigram ("A B" → "C")   ← recommended default
  // 3 = 4-gram  ("A B C" → "D") ← needs large corpus
  order int = 2
  // smoothing is the Laplace (additive) smoothing coefficient.
  // 0.0  = no smoothing (exact counts)
  // 0.01 = small nudge that prevents zero probabilities for unseen n-grams
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

// new returns an empty Markov model with the given config.
pub fn new(cfg Config) Markov {
  return Markov{
    model: map[string]map[string]f64{}
    cfg: cfg
  }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

// build_from_tokens trains the model from a pre-tokenized slice.
// This is the core builder; all other build_* functions call this.
pub fn build_from_tokens(tokens []string, cfg Config) Markov {
  mut m := new(cfg)
  order := cfg.order
  if order < 1 || tokens.len <= order {
    return m
  }

  m.tok_count = tokens.len

  // --- Count phase ---
  mut counts := map[string]map[string]int{}
  for i in 0 .. tokens.len - order {
    ctx := join_ctx(tokens[i..i + order])
    nxt := tokens[i + order]
    if ctx !in counts {
      counts[ctx] = map[string]int{}
    }
    counts[ctx][nxt] = (counts[ctx][nxt] or { 0 }) + 1
  }

  // --- Probability phase (with optional Laplace smoothing) ---
  vocab_size := if cfg.smoothing > 0.0 { tokens.len } else { 0 }

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

// build_from_text tokenizes `text` then trains the model.
pub fn build_from_text(text string, cfg Config) Markov {
  return build_from_tokens(tokenize(text), cfg)
}

// build_from_file reads a corpus file and trains the model.
pub fn build_from_file(path string, cfg Config) !Markov {
  raw := os.read_file(path) or { return error('cannot read file: ${err}') }
  return build_from_text(raw, cfg)
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

// save writes the model (plus metadata) to a JSON file.
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

// load reads a model previously saved with save().
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
    model: model
    cfg: cfg
    tok_count: tok_count
  }
}

// ---------------------------------------------------------------------------
// Querying
// ---------------------------------------------------------------------------

// get_next_probabilities returns the raw probability distribution for a
// context string (tokens already joined with the internal separator).
pub fn (m Markov) get_next_probabilities(ctx string) map[string]f64 {
  if ctx in m.model {
    return m.model[ctx].clone()
  }
  return map[string]f64{}
}

// get_next_probs_for_tokens is the ergonomic wrapper: pass the last N tokens
// as a slice and get back the probability map.
pub fn (m Markov) get_next_probs_for_tokens(ctx_tokens []string) map[string]f64 {
  return m.get_next_probabilities(join_ctx(ctx_tokens))
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// sample picks a token from a probability distribution.
//
// temperature controls randomness:
//   - 1.0 → unchanged distribution
//   - < 1.0 → sharper / more deterministic  (e.g. 0.5)
//   - > 1.0 → flatter / more creative       (e.g. 1.5)
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

  // Fallback: highest-probability token
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

// GenerateConfig holds options for text generation.
pub struct GenerateConfig {
pub:
  max_tokens  int = 100
  temperature f64 = 1.0
  // back_off controls whether shorter contexts are tried when the full
  // context has no match. Highly recommended to keep true.
  back_off bool = true
}

// generate returns a slice of generated tokens given a list of seed tokens.
//
// Strategy:
//  1. Try the full-order context.
//  2. If back_off is true and no match, try progressively shorter contexts.
//  3. Stop when max_tokens is reached or no continuation can be found.
pub fn (m Markov) generate(seed []string, gcfg GenerateConfig) []string {
  order := m.cfg.order
  mut history := seed.clone()

  // Pad history so we always have `order` tokens of context
  for history.len < order {
    history.insert(0, '')
  }

  mut output := []string{}

  for _ in 0 .. gcfg.max_tokens {
    mut picked := ''

    if gcfg.back_off {
      // Try longest context first, shorten on miss
      for ctx_len := order; ctx_len >= 1; ctx_len-- {
        start := history.len - ctx_len
        ctx := join_ctx(history[start..])
        probs := m.get_next_probabilities(ctx)
        if probs.len > 0 {
          picked = sample(probs, gcfg.temperature)
          break
        }
      }
    } else {
      ctx := join_ctx(history[history.len - order..])
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

// generate_text runs generate() and joins the result into a readable string,
// re-attaching punctuation without leading spaces.
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

// stats returns a summary string about the model.
pub fn (m Markov) stats() string {
  return 'Markov(order=${m.cfg.order}, smoothing=${m.cfg.smoothing}, ' +
    'contexts=${m.model.len}, tok_count=${m.tok_count})'
}

struct Pair {
  token string
  prob  f64
}

// top_continuations returns the top-N most likely next tokens for a context.
pub fn (m Markov) top_continuations(ctx_tokens []string, n int) []string {
  probs := m.get_next_probs_for_tokens(ctx_tokens)
  mut pairs := []Pair{}
  for tok, prob in probs {
    pairs << Pair{
      token: tok
      prob:prob
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