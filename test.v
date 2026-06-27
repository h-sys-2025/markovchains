module main

import h_sys_2025.vmarkov.markovchains as markov

fn main() {
	// ── 1. Train on text ─────────────────────────────────────────────────────
	m := markov.from_file('./raw_data.txt', markov.cfg()) or { panic(err) }
	println(m.stats())

	// ── 2. One-liner API ─────────────────────────────────────────────────────
	println(m.next('the quick'))              // single next token
	println(m.next_n('the', 5))              // 5 independent next tokens
	println(m.complete('the quick brown', 30)) // seed + 30 generated tokens

	// ── 3. Full control ──────────────────────────────────────────────────────
	result := m.generate_text('hello world', markov.GenConfig{
		max_tokens:  40
		temperature: 0.8    // 0.5 focused · 1.0 normal · 1.5 creative
		top_k_n:     10     // only sample from top-10 tokens
		top_p_val:   0.9    // or: nucleus sampling (pick one or combine)
		stop_token:  '.'    // stop at first full stop
		back_off:    true
	})
	println(result)

	// ── 4. Inspect predictions ───────────────────────────────────────────────
	preds := m.top(['the', 'quick'], 5)
	for p in preds { println(p) }             // prints: "brown" (0.4123) etc.

	// ── 5. Probability of a specific next token ──────────────────────────────
	p := m.prob(['hello'], 'world')
	println('P(world | hello) = ${p:.4f}')

	// ── 6. Random seed when you have no prompt ───────────────────────────────
	start := m.random_start()
	println(m.complete(start.join(' '), 20))

	// ── 7. Character-level model (name / word generation) ────────────────────
	char_m := markov.from_file_chars('./raw_data.txt', markov.cfg_order(3)) or { panic(err) }
	println(char_m.generate_chars('ma', markov.GenConfig{ max_tokens: 10 }))

	// ── 8. Works on anything — DNA sequences ─────────────────────────────────
	dna_tokens := markov.split_by('ATG CGT ATG AAA CGT TTT ATG CGT AAA', ' ')
	dna_m := markov.from_tokens(dna_tokens, markov.cfg_order(1))
	println(dna_m.walk(['ATG'], 8))           // e.g. [CGT, ATG, AAA, CGT, ...]

	// ── 9. Merge two models ──────────────────────────────────────────────────
	m2 := markov.from_file('./extra_data.txt', markov.cfg()) or { m } // fallback to m
	merged := m.merge(m2)
	println(merged.stats())

	// ── 10. Add more data to an existing model ───────────────────────────────
	updated := m.train_more('extra sentence to fold in without retraining from scratch.')
	println(updated.stats())

	// ── 11. Model quality on held-out text ───────────────────────────────────
	ppl := m.perplexity('the quick brown fox jumps over the lazy dog')
	println('perplexity: ${ppl:.2f}')   // lower = model fits this text better

	// ── 12. Save & load ──────────────────────────────────────────────────────
	m.save('./model.json') or { println('save failed: ${err}') }
	loaded := markov.load('./model.json') or { panic(err) }
	println(loaded.stats())
}