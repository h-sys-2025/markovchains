module main

import h_sys_2025.vmarkov.markovchains as markov

fn main() {
  m := markov.from_file("./raw_data.txt", markov.cfg()) or { panic(err) }
  println(m.stats())

  println(m.next("WAR IS"))              // single next token
  println(m.next_n("IGNORANCE", 5))              // 5 independent next tokens
  println(m.complete("The Ministry of Truth", 30)) // seed + 30 generated tokens

  result := m.generate_text("hello world", markov.GenConfig{
    max_tokens:  40
    temperature: 0.5    // 0.5 focused · 1.0 normal · 1.5 creative
    top_k_n:     5     // only sample from top-10 tokens
    top_p_val:   0.9    // or: nucleus sampling (pick one or combine)
    stop_token:  "."    // stop at first full stop
    back_off:    true
  })
  println(result)

  preds := m.top(["WAR", "IS"], 5)
  for p in preds { println(p) }             // prints: "brown" (0.4123) etc.

  p := m.prob(["IS"], "STRENGHT")
  println("P(IS | STRENGHT) = ${p}")

  start := m.random_start()
  println(m.complete(start.join(" "), 30))

  char_m := markov.from_file_chars("./raw_data.txt", markov.cfg_order(3)) or { panic(err) }
  println(char_m.generate_chars("big ", markov.GenConfig{ max_tokens: 6 }))

  dna_tokens := markov.split_by("ATG CGT ATG AAA CGT TTT ATG CGT AAA", " ")
  dna_m := markov.from_tokens(dna_tokens, markov.cfg_order(1))
  println(dna_m.walk(["ATG"], 8))           // e.g. [CGT, ATG, AAA, CGT, ...]

  m2 := markov.from_file("./extra_data.txt", markov.cfg()) or { m } // fallback to m
  merged := m.merge(m2)
  println(merged.stats())

  updated := m.train_more("extra sentence to fold in without retraining from scratch.")
  println(updated.stats())

  ppl := m.perplexity("the quick brown fox jumps over the lazy dog")
  println("perplexity: ${ppl:.2f}")   // lower = model fits this text better

  m.save("./model.json") or { println("save failed: ${err}") }
  loaded := markov.load("./model.json") or { panic(err) }
  println(loaded.stats())
}