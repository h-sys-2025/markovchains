# markovchains in v.
- A simple lib to make and experiment with markov chains, all in v, minimal+lightweight.

![0.29](https://img.shields.io/badge/version-0.29-white?style=flat)
![GitHub](https://img.shields.io/badge/license-MIT-blue?style=flat)
![vlang](http://img.shields.io/badge/V-0.5+-%236d8fc5?style=flat)

## Installazation:
```sh
v install h-sys-2025.vmarkov
```

## Example:
```v
module main

import markovchains as markov

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
```
