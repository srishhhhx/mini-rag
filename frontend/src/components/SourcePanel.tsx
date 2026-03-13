'use client'

import { SourceChunk } from '@/lib/api'

interface SourcePanelProps {
  sources: SourceChunk[]
  activePill: number | null
}

export default function SourcePanel({ sources, activePill }: SourcePanelProps) {
  const displayed = activePill !== null ? [sources[activePill]] : sources

  return (
    <div className="source-panel">
      <p className="source-panel__title">
        {activePill !== null
          ? `Source · page ${sources[activePill]?.page}`
          : `${sources.length} source${sources.length !== 1 ? 's' : ''} retrieved`}
      </p>
      {displayed.map((chunk, i) => (
        <div key={i} className="source-chunk">
          <div className="source-chunk__meta">
            <span className="source-chunk__page">p.{chunk.page}</span>
            <span className="source-chunk__type">{chunk.chunk_type}</span>
            {chunk.section_header && (
              <span
                style={{
                  fontSize: '0.68rem',
                  color: 'var(--text-muted)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  maxWidth: 160,
                }}
              >
                {chunk.section_header}
              </span>
            )}
            <span className="source-chunk__score">
              {(chunk.score * 100).toFixed(0)}%
            </span>
          </div>
          <p className="source-chunk__text">{chunk.text}</p>
        </div>
      ))}
    </div>
  )
}
