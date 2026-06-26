O problema não é tanto o prompt, mas sim que você está pedindo uma tarefa muito subjetiva ("fit the allotted time constraint"). Um modelo de 7B tende a não raciocinar numericamente sobre comprimento de texto. Ele entende muito melhor **regras explícitas e mensuráveis**.

Para dublagem eu inverteria a abordagem: em vez de dizer *"fit the allotted time constraint"*, eu diria exatamente quantos caracteres e palavras ele pode usar, além da prioridade de cada objetivo.

Por exemplo, para a etapa de **reescrita**, eu usaria algo assim:

```text
You are a professional Brazilian Portuguese dubbing adapter.

Your task is to rewrite a Portuguese subtitle so it sounds natural in spoken Brazilian Portuguese.

Priorities (highest to lowest):
1. Preserve the original meaning.
2. Preserve the original tone.
3. Make it sound like native speech.
4. Stay within the maximum length.

Rules:
- Maximum characters: {max_chars}
- Maximum words: {max_words}
- Never exceed either limit.
- If necessary, sacrifice detail instead of meaning.
- Prefer shorter and more common words.
- Remove redundancy.
- Merge ideas naturally.
- Do not add information.
- Do not explain.
- Output only the rewritten subtitle.
- Output strictly in Brazilian Portuguese.
```

E para a etapa de **condensação**:

```text
You are a professional subtitle condenser for Brazilian Portuguese dubbing.

Rewrite the subtitle using fewer words while preserving the essential meaning.

Rules:
- Maximum characters: {max_chars}
- Maximum words: {max_words}
- Every word must justify its existence.
- Remove adjectives, filler words and repeated ideas first.
- Prefer active voice.
- Prefer common vocabulary.
- Keep the sentence easy to pronounce aloud.
- Never exceed the limits.
- Output only the condensed subtitle.
- Output strictly in Brazilian Portuguese.
```

---

## Melhor ainda: forneça a velocidade de fala

Em vez de apenas passar `max_chars`, passe também:

```
Duration: 4.2 seconds
Maximum characters: 63
Target speech rate: about 2.5 words/second
```

Os modelos seguem melhor isso do que uma instrução genérica.

---

## Poucos exemplos (few-shot) ajudam muito

Qwen 2.5 melhora bastante quando vê exemplos.

```
Example

Original:
Aquele que governa o mundo e acredita plenamente que os deuses estão pessoalmente investidos na sua propriedade imperial.

Limit:
60 characters

Good:
Quem governa o mundo crê que os deuses protegem seu império.

Bad:
Quem governa o mundo acredita plenamente que os deuses...
```

Ou

```
Original:
It was believed that the sacred fire protected Rome.

Translation:
Acreditava-se que o fogo sagrado protegia Roma.

Limit:
34 chars

Good:
O fogo sagrado protegia Roma.
```

Dois ou três exemplos costumam aumentar bastante a qualidade.

---

## Eu faria em duas etapas

Em vez de apenas "rewrite" e "condense", faria um pipeline:

```
EN
↓
Translate literally
↓
Naturalize PT-BR
↓
Fit duration
```

Na última etapa, o prompt seria extremamente específico:

```
Your only goal is to fit the subtitle into the limit.

Allowed operations:
- remove redundant words
- replace long words with shorter synonyms
- restructure the sentence
- omit secondary information

Forbidden:
- change the main meaning
- invent information

Return ONLY the subtitle.
```

---

## Outra observação importante

Você mencionou **15 caracteres/s**.

Na prática de legendagem e dublagem, normalmente funciona melhor limitar por **sílabas**, não por caracteres.

Por exemplo:

```
"acredita plenamente"

↓

"crê"
```

reduz de 21 caracteres para apenas 3, mas também reduz drasticamente o número de sílabas, que é o que realmente afeta o tempo de fala.

Uma métrica mais robusta seria algo como:

```
score =
0.6 * syllable_count +
0.3 * word_count +
0.1 * character_count
```

ou até usar apenas sílabas se você dispõe de um silabificador em português. A correlação com a duração da fala é significativamente melhor do que caracteres.

---

## Se fosse desenvolver esse pipeline hoje

Considerando seus projetos anteriores com tradução, XTTS e sincronização de áudio, eu faria a IA trabalhar com um prompt estruturado como este:

```
ORIGINAL (English):
...

CURRENT TRANSLATION (Portuguese):
...

TARGET DURATION:
4.2 s

MAX CHARACTERS:
63

MAX WORDS:
11

STYLE:
Documentary narration

Task:
Rewrite the Portuguese subtitle so that:
- it sounds like spoken Brazilian Portuguese;
- preserves the original meaning and tone;
- fits within BOTH limits;
- is easy to dub naturally.

Return ONLY the rewritten subtitle.
```

Esse formato costuma produzir resultados mais consistentes em modelos da faixa de 7B, porque reduz ambiguidades e fornece restrições objetivas. Além disso, eu consideraria calcular o limite de palavras a partir da duração (≈2,5 palavras/s) e usar o limite de caracteres apenas como uma restrição secundária.
