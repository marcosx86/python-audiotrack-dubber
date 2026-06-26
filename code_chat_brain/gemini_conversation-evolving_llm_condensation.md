Essa é uma excelente iniciativa, Marcos. O maior desafio na dublagem e na criação de *script sync* (especialmente do inglês para o português) é que o português brasileiro costuma ser entre **20% e 30% mais longo** em número de sílabas e caracteres para expressar a mesma ideia.

Seus prompts atuais estão bem estruturados na teoria, mas modelos da família Qwen (e LLMs em geral) tendem a falhar em tarefas de contagem estrita de caracteres/palavras porque eles pensam em **tokens** (pedaços de palavras), e não em letras ou fonemas. Além disso, o prompt em inglês pode fazer com que o modelo gaste "atenção" traduzindo o comando internamente em vez de focar na métrica.

Para fazer o `qwen2.5-7b` respeitar rigorosamente a métrica de **15 caracteres por segundo (CPS)** ou **2.5 palavras por segundo (PPS)**, precisamos mudar a estratégia de instrução.

---

## 1. Otimização dos Prompts (Exemplo Prático)

Em vez de dar comandos genéricos como *"obey the character limit"*, você deve passar o cálculo pronto para o modelo no *user prompt* (ex: "Limite máximo: X caracteres") e treinar o *system prompt* para ser implacável com o tamanho. Passar o prompt do sistema em português também ajuda o modelo de 7B a se manter no mesmo "trilho" de raciocínio.

Aqui estão duas sugestões reformuladas:

### Opção 1: Foco em Reescrita Natural (Dublagem Adaptada)

> **System Prompt:**
> `Você é um roteirista de dublagem profissional e especialista em sincronismo labial e tempo de fala. Sua função é reescrever a frase em português enviada pelo usuário para que ela caiba estritamente no limite de caracteres fornecido. É vital manter o sentido original, o tom e a emoção, mas você deve cortar palavras redundantes e usar sinônimos mais curtos para garantir uma dicção fluida de 15 caracteres por segundo. Regra estrita: responda APENAS com o texto reescrito, sem aspas, sem explicações e sem notas de tradução.`

### Opção 2: Foco em Condensação Máxima (Frases Curtas e Rápidas)

> **System Prompt:**
> `Você é um editor de dublagem focado em síntese e velocidade de fala. Reduza drasticamente a frase em português enviada pelo usuário para torná-la o mais curta, direta e rápida de falar possível, sem perder o significado central. Respeite obrigatoriamente o limite de caracteres informado. Remova qualquer floreio linguístico. Regra estrita: responda APENAS com o texto condensado, sem introduções, sem aspas e sem justificativas.`

---

## 2. A Estratégia no Código (Python)

Como você está usando a integração da OpenAI com o LM Studio, o segredo para isso funcionar 100% não está apenas no *System Prompt*, mas em como você envia os dados no *User Prompt*.

Você precisa calcular o tempo da frase original (via áudio ou SRT) e injetar o limite exato de caracteres no comando.

Veja este exemplo de estrutura de automação em Python:

```python
import math


def adaptar_legenda_para_dublagem(texto_traduzido, duracao_segundos, client):
    # Calcula o limite estrito com base na métrica de 15 caracteres por segundo
    limite_caracteres = math.floor(duracao_segundos * 15)

    # Monta o prompt do usuário injetando a métrica que o modelo precisa seguir
    user_content = (
        f"Texto original: '{texto_traduzido}'\n"
        f"LIMITE MÁXIMO ABSOLUTO: {limite_caracteres} caracteres (incluindo espaços).\n"
        f"Reescreva respeitando o limite."
    )

    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct-uncensored",
        messages=[
            {
                "role": "system",
                "content": "Você é um roteirista de dublagem profissional... [Use o prompt adaptado aqui]",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,  # Temperatura baixa para o modelo ser mais factual e obediente
        max_tokens=100,
    )

    texto_gerado = response.choices[0].message.content.strip()

    # Sanitização extra caso o modelo decida colocar aspas por conta própria
    if texto_gerado.startswith('"') and texto_gerado.endswith('"'):
        texto_gerado = texto_gerado[1:-1]

    return texto_gerado

```

---

## 3. Dicas Extras para Modelos Locais (7B)

* **Temperatura Baixa:** Defina a `temperature` entre `0.2` e `0.4`. Valores mais altos fazem o modelo "viajar" e ignorar os limites de caracteres que você impôs.
* **Parâmetro de Penalidade de Presença/Repetição:** Se o modelo começar a repetir palavras para tentar preencher espaço, ajuste o `repetition_penalty` no LM Studio para algo em torno de `1.1`.
* **Validação em Python (Loop de Feedback):** Como o Qwen de 7B pode eventualmente estourar o limite por 2 ou 3 caracteres, você pode criar uma validação simples no seu script: se `len(texto_gerado) > limite_caracteres`, o Python faz uma nova requisição automática (um *retry*) adicionando um aviso: *"Você ultrapassou o limite anterior. Refaça com menos de X caracteres"*.

Com o cálculo do limite de caracteres sendo feito dinamicamente pelo Python e injetado diretamente no input do usuário, o modelo terá muito mais facilidade para acertar o tempo de dicção do seu gerador de áudio (TTS).

Se precisar ajustar a lógica do script de validação ou do cálculo de tempo, me avise!