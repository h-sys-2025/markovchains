module main

import markovchains as markov

fn main() {
    // Option 1: Build from text
    text := "hello world this is a test hello again world"
    m := markov.build_from_text(text, 1)

    // Option 2: Load from JSON (same format as your original)
    // m := markov.load("./v1.json") or { panic(err) }

    println("Model built with ${m.model.len} states")

    // Generate
    prompt := "hello"
    result := m.generate_text(prompt, 50)
    println("${prompt} ${result}")

    // Save model
    m.save("model.json") or { println("Save failed: ${err}") }
}