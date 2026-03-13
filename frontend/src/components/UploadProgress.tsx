'use client'

interface UploadProgressProps {
  filename: string
  error: string | null
  onRetry: () => void
}

export default function UploadProgress({ filename, error, onRetry }: UploadProgressProps) {
  return (
    <div className="processing-screen">
      <div className="processing-card">
        {error ? (
          <>
            <div style={{ fontSize: '1.5rem' }}>⚠️</div>
            <h2 className="processing-card__title">Ingestion failed</h2>
            <p className="processing-error">{error}</p>
            <button className="btn-retry" onClick={onRetry}>Try another PDF</button>
          </>
        ) : (
          <>
            <div className="spinner" />
            <h2 className="processing-card__title">
              Indexing<span className="dots" />
            </h2>
            <p className="processing-card__filename">{filename}</p>
            <p className="processing-card__status">
              Parsing · Chunking · Embedding
            </p>
          </>
        )}
      </div>
    </div>
  )
}
