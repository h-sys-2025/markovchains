module main

import h_sys_2025.vmarkov.markovchains as markov

fn main() {
    text := "hello world this is a test hello again world"
    m := markov.build_from_text(text, 1)

    println("Model built with ${m.model.len} states")

    // generation/compeltion.
    prompt := "hello"
    result := m.generate_text(prompt, 50)
    println("${prompt} ${result}")

    // save/load model:
    // // load:
    // m := markov.load("./v1.json") or { panic(err) }
    // // save:
    // m.save("model.json") or { println("Save failed: ${err}") }
}