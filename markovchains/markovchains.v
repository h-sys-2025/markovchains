module markovchains

import os
import x.json2
import rand

pub struct Markov {
pub mut:
    model map[string]map[string]f64
}

// new creates an empty Markov model
pub fn new() Markov {
    return Markov{
        model: map[string]map[string]f64{}
    }
}

// build_from_text builds a Markov model from training text (order=1 = bigram)
pub fn build_from_text(text string, order int) Markov {
    mut m := new()
    if order < 1 {
        return m
    }

    words := text.to_lower().split(' ').filter(it.len > 0)
    if words.len <= order {
        return m
    }

    for i in 0 .. (words.len - order) {
        mut key := ''
        for j in 0 .. order {
            if j > 0 {
                key += ' '
            }
            key += words[i + j]
        }

        next := words[i + order]

        if key !in m.model {
            m.model[key] = map[string]f64{}
        }
        m.model[key][next] = m.model[key][next] + 1.0
    }

    // Convert counts to probabilities
    for _, mut next_map in m.model {
        mut total := 0.0
        for _, count in next_map {
            total += count
        }
        if total > 0 {
            for next_word, count in next_map {
                next_map[next_word] = count / total
            }
        }
    }

    return m
}

// load loads model from JSON file
pub fn load(path string) !Markov {
    json_text := os.read_file(path)!
    root := json2.decode[json2.Any](json_text) or { return error('Failed to decode JSON: ${err}') }

    data := root.as_map()
    mut model := map[string]map[string]f64{}

    for key, value in data {
        inner := value.as_map()
        mut inner_map := map[string]f64{}
        for next_word, prob_any in inner {
            inner_map[next_word] = prob_any.f64()
        }
        model[key] = inner_map.clone()  // Fixed: explicit clone
    }

    return Markov{
        model: model
    }
}

// save writes the model to JSON
pub fn (m Markov) save(path string) ! {
    mut root := map[string]json2.Any{}
    for key, next_map in m.model {
        mut inner := map[string]json2.Any{}
        for next, prob in next_map {
            inner[next] = prob
        }
        root[key] = inner
    }

    json_str := json2.encode(root)
    os.write_file(path, json_str)!
}

// get_next_probabilities returns probabilities for next token
pub fn (m Markov) get_next_probabilities(current string) map[string]f64 {
    if current in m.model {
        return m.model[current].clone()
    }
    return map[string]f64{}
}

fn pick_weighted(probs map[string]f64) string {
    if probs.len == 0 {
        return ''
    }

    mut total_weight := 0.0
    for _, w in probs {
        total_weight += w
    }

    if total_weight <= 0.0 {
        return ''
    }

    r := rand.f64() * total_weight
    mut cumulative := 0.0

    for token, weight in probs {
        cumulative += weight
        if r <= cumulative {
            return token
        }
    }

    return probs.keys()[0]
}

// generate returns list of generated tokens
pub fn (m Markov) generate(prompt string, max_tokens int) []string {
    if max_tokens <= 0 {
        return []
    }

    tokens := prompt.to_lower().split(' ').filter(it.len > 0)
    if tokens.len == 0 {
        return []
    }

    mut current := tokens[tokens.len - 1]
    mut result := []string{}

    for _ in 0 .. max_tokens {
        probs := m.get_next_probabilities(current)
        if probs.len == 0 {
            break
        }

        next := pick_weighted(probs)
        if next == '' {
            break
        }

        result << next
        current = next
    }

    return result
}

// generate_text returns generated text as string
pub fn (m Markov) generate_text(prompt string, max_tokens int) string {
    tokens := m.generate(prompt, max_tokens)
    return tokens.join(' ')
}