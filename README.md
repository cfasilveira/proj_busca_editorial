# editorial

Aplicativo genérico e leve para **coleta, processamento e análise de textos**
(editoriais, artigos, entrevistas transcritas), capaz de gerar **perfis
linguísticos e ideológicos** com transparência e auditoria.

Modular, reutilizável em diferentes domínios (financeiro, político,
acadêmico), com logs estruturados e validação estatística.

## Stack

| Camada          | Tecnologia                                        |
| --------------- | ------------------------------------------------- |
| Linguagem       | Python >= 3.12                                    |
| Gerenciamento   | [uv](https://github.com/astral-sh/uv)             |
| NLP             | spaCy (modelo pt) → fallback NLTK → tokenizador interno |
| Vetorial        | FAISS (local), interface trocável (Milvus/Pinecone) |
| Embeddings      | TF-IDF + TruncatedSVD (leve e offline)            |
| Logs            | Estruturados em JSON e console                    |
| Isolamento      | Docker                                            |

## Instalação

```bash
uv sync                # ambiente + dependências
uv run editorial --help
```

### Modelo spaCy (opcional, melhora tokenização)

```bash
uv run python -m spacy download pt_core_news_sm
```

### Auditoria de dependências (extra opcional)

```bash
uv sync --extra audit
uv run editorial audit
```

## Uso

```bash
# Pipeline completo (CSV → perfil → índice → relatório JSON)
uv run editorial pipeline data/samples/editorials.csv --outdir reports

# Consulta vetorial no corpus indexado
uv run editorial search "responsabilidade fiscal" --topk 3

# Auditoria de dependências
uv run editorial audit

# Validar token de acesso
uv run editorial auth-check SEU_TOKEN --scope read

# Logs em JSON (flag global antes do subcomando)
uv run editorial --log-json pipeline data/samples/editorials.csv
```

## Arquitetura modular

```
src/editorial/
├── config.py              # configuração central (env / .env)
├── logging_setup.py       # logs JSON + console
├── errors.py              # exceções tipadas com mensagens amigáveis
├── pipeline.py            # orquestração dos estágios
├── cli.py                 # interface de linha de comando
├── ingestion/             # CSV, API (httpx), scraping (BeautifulSoup)
├── processing/            # tokenização, limpeza, normalização estatística
├── scientific/            # matriz de decisão, regressão, análise bayesiana
├── vector/                # embeddings (TF-IDF+SVD) + store FAISS
├── reports/               # relatórios JSON/CSV (+ stub PDF)
└── security/              # auditoria de dependências, controle de acesso
```

## Filosofia de tratamento de erros

- **Early return / Fail first**: entradas inválidas abortam no início
  (ex.: CSV vazio → `IngestionError`).
- **Fail gracefully**: erros são registrados em log estruturado e o usuário
  recebe mensagem clara (`user_message`), sem travar o sistema.

## Docker

```bash
docker build -t editorial .
mkdir -p reports && chmod 777 reports   # volume precisa ser gravável pelo usuário do container (uid 10001)
docker run --rm -v "$PWD/reports:/app/reports" editorial pipeline data/samples/editorials.csv --outdir /app/reports
```

## Testes

```bash
uv run pytest -q
```
