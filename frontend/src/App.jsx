import { useEffect, useMemo, useState } from 'react'

function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState('http://127.0.0.1:8000')
  const [collectionName, setCollectionName] = useState('pdf_embeddings')
  const [filePath, setFilePath] = useState('')
  const [maxVectors, setMaxVectors] = useState('')
  const [targetDimension, setTargetDimension] = useState('')
  const [embedModelKeys, setEmbedModelKeys] = useState(['1', '2'])

  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(4)
  const [llmModelKeys, setLlmModelKeys] = useState(['1', '2', '3'])

  const [ingestResult, setIngestResult] = useState(null)
  const [queryResult, setQueryResult] = useState(null)
  const [availableModels, setAvailableModels] = useState({
    llm_models: {},
    embedding_models: {},
  })

  const [isIngesting, setIsIngesting] = useState(false)
  const [isQuerying, setIsQuerying] = useState(false)
  const [error, setError] = useState('')

  const endpoint = useMemo(() => apiBaseUrl.replace(/\/$/, ''), [apiBaseUrl])

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${endpoint}/models`)
        if (!res.ok) {
          throw new Error(`Failed to fetch model options (${res.status})`)
        }
        const data = await res.json()
        setAvailableModels(data)
      } catch (err) {
        setError(err.message)
      }
    }

    fetchModels()
  }, [endpoint])

  const toggleSelection = (key, selected, setSelected) => {
    if (selected.includes(key)) {
      setSelected(selected.filter((k) => k !== key))
      return
    }
    setSelected([...selected, key])
  }

  const ingestDocument = async (event) => {
    event.preventDefault()
    setError('')
    setIsIngesting(true)
    setIngestResult(null)

    try {
      const payload = {
        file_path: filePath,
        collection_name: collectionName,
        embedding_model_keys: embedModelKeys,
        max_vectors: maxVectors ? Number(maxVectors) : null,
        target_dimension: targetDimension ? Number(targetDimension) : null,
      }

      const res = await fetch(`${endpoint}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Ingestion failed')
      }

      setIngestResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsIngesting(false)
    }
  }

  const askQuery = async (event) => {
    event.preventDefault()
    setError('')
    setIsQuerying(true)
    setQueryResult(null)

    try {
      const payload = {
        query,
        collection_name: collectionName,
        top_k: Number(topK),
        llm_model_keys: llmModelKeys,
      }

      const res = await fetch(`${endpoint}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Query failed')
      }

      setQueryResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <main className="app-shell">
      <div className="bg-orb orb-a" aria-hidden="true" />
      <div className="bg-orb orb-b" aria-hidden="true" />

      <header className="hero-header">
        <p className="kicker">RAG Control Room</p>
        <h1>Document Intelligence Studio</h1>
        <p className="subtitle">
          Ingest PDFs, search vectors, and compare multi-LLM answers in one live dashboard.
        </p>
      </header>

      <section className="api-bar">
        <label htmlFor="api-base">Backend URL</label>
        <input
          id="api-base"
          value={apiBaseUrl}
          onChange={(e) => setApiBaseUrl(e.target.value)}
          placeholder="http://127.0.0.1:8000"
        />

        <label htmlFor="collection">Collection</label>
        <input
          id="collection"
          value={collectionName}
          onChange={(e) => setCollectionName(e.target.value)}
          placeholder="pdf_embeddings"
        />
      </section>

      {error ? <div className="alert">{error}</div> : null}

      <section className="panel-grid">
        <form className="panel" onSubmit={ingestDocument}>
          <h2>Ingest Document</h2>
          <p className="panel-note">Load PDF, create embeddings, and store vectors to Chroma.</p>

          <label htmlFor="pdf-path">PDF file path</label>
          <input
            id="pdf-path"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="C:/docs/your-file.pdf"
            required
          />

          <div className="inline-grid">
            <div>
              <label htmlFor="max-vectors">Max vectors</label>
              <input
                id="max-vectors"
                type="number"
                min="1"
                value={maxVectors}
                onChange={(e) => setMaxVectors(e.target.value)}
                placeholder="All"
              />
            </div>

            <div>
              <label htmlFor="target-dim">Target dimension</label>
              <input
                id="target-dim"
                type="number"
                min="1"
                value={targetDimension}
                onChange={(e) => setTargetDimension(e.target.value)}
                placeholder="Original"
              />
            </div>
          </div>

          <p className="chip-title">Embedding models</p>
          <div className="chip-wrap">
            {Object.entries(availableModels.embedding_models).map(([key, name]) => (
              <button
                type="button"
                key={key}
                className={`chip ${embedModelKeys.includes(key) ? 'active' : ''}`}
                onClick={() => toggleSelection(key, embedModelKeys, setEmbedModelKeys)}
              >
                {key} - {name}
              </button>
            ))}
          </div>

          <button className="cta" type="submit" disabled={isIngesting}>
            {isIngesting ? 'Ingesting...' : 'Create and Store Embeddings'}
          </button>

          {ingestResult ? (
            <div className="result-box">
              <p>{ingestResult.message}</p>
              <p>Vectors stored: {ingestResult.vectors_stored}</p>
              <p>Final dimension: {ingestResult.final_dimension}</p>
            </div>
          ) : null}
        </form>

        <form className="panel" onSubmit={askQuery}>
          <h2>Query + Compare Models</h2>
          <p className="panel-note">Retrieve relevant chunks and ask selected LLMs.</p>

          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question grounded in your document"
            rows={5}
            required
          />

          <label htmlFor="top-k">Top K chunks</label>
          <input
            id="top-k"
            type="number"
            min="1"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />

          <p className="chip-title">LLM models</p>
          <div className="chip-wrap">
            {Object.entries(availableModels.llm_models).map(([key, name]) => (
              <button
                type="button"
                key={key}
                className={`chip ${llmModelKeys.includes(key) ? 'active' : ''}`}
                onClick={() => toggleSelection(key, llmModelKeys, setLlmModelKeys)}
              >
                {key} - {name}
              </button>
            ))}
          </div>

          <button className="cta" type="submit" disabled={isQuerying}>
            {isQuerying ? 'Searching and Asking...' : 'Run RAG Query'}
          </button>
        </form>
      </section>

      {queryResult ? (
        <section className="results-grid">
          <article className="panel">
            <h2>Retrieved Chunks</h2>
            <div className="chunk-list">
              {queryResult.retrieved_chunks?.length ? (
                queryResult.retrieved_chunks.map((chunk, idx) => (
                  <div className="chunk-card" key={`${idx}-${chunk.slice(0, 16)}`}>
                    <p className="chunk-index">Chunk {idx + 1}</p>
                    <p>{chunk}</p>
                  </div>
                ))
              ) : (
                <p>No chunks found.</p>
              )}
            </div>
          </article>

          <article className="panel">
            <h2>LLM Answers</h2>
            <div className="answer-list">
              {Object.entries(queryResult.answers || {}).map(([model, answer]) => (
                <div className="answer-card" key={model}>
                  <p className="model-name">{model}</p>
                  <p>{answer}</p>
                </div>
              ))}
            </div>
          </article>
        </section>
      ) : null}
    </main>
  )
}

export default App
