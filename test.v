module main

import h_sys_2025.vmarkov.markovchains as markov

fn main() {
	cfg := markov.Config{
		order:     2
		smoothing: 0.01
	}

	m := markov.build_from_file('./raw_data.txt', cfg) or { panic(err) }
	println('Model built -- ${m.stats()}')

	prompt := 'hello'
	gcfg := markov.GenerateConfig{
		max_tokens:  1
		temperature: 1.0
		back_off:    true
	}

	result := m.generate_text(prompt, gcfg)
	println('Generated: ${result}')

	tops := m.top_continuations(['death'], 3)
	println('Top continuations after "death": ${tops}')

	m.save('./model.json') or { println('Save failed: ${err}') }

	// m2 := markov.load('./model.json') or { panic(err) }
	// println('Loaded -- ${m2.stats()}')
}