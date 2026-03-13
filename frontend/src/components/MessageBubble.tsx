'use client'

import { Message } from '@/hooks/useChat'
import SourcePanel from './SourcePanel'

interface MessageBubbleProps {
  message: Message
  onToggleSources: () => void
  onPillClick: (idx: number) => void
}

export default function MessageBubble({ message, onToggleSources, onPillClick }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const isLowConf = !message.confident && !message.streaming

  return (
    <div className={`message message--${isUser ? 'user' : 'ai'}${isLowConf ? ' message--low-confidence' : ''}`}>
      <span className="message__role">{isUser ? 'You' : 'Assistant'}</span>

      <div className="message__bubble">
        {message.content}
        {message.streaming && <span className="streaming-cursor" />}
      </div>

      {/* Source pills + low-confidence badge (AI only, once done streaming) */}
      {!isUser && !message.streaming && (
        <>
          {isLowConf && (
            <div className="low-confidence-badge">
              <span>⚠</span>
              <span>Answer not found in document</span>
            </div>
          )}

          {message.sources.length > 0 && (
            <div className="message__sources">
              {message.sources.map((src, i) => (
                <button
                  key={i}
                  className={`source-pill${message.activePill === i ? ' active' : ''}`}
                  onClick={() => onPillClick(i)}
                  aria-label={`Source page ${src.page}`}
                  title={`Page ${src.page} — ${src.chunk_type}`}
                >
                  p.{src.page}
                </button>
              ))}
              <button
                className={`source-pill${message.showSources && message.activePill === null ? ' active' : ''}`}
                onClick={onToggleSources}
                aria-label="Show all sources"
              >
                {message.showSources && message.activePill === null ? 'Hide sources' : 'All sources'}
              </button>
            </div>
          )}
        </>
      )}

      {/* Source panel */}
      {!isUser && message.showSources && message.sources.length > 0 && (
        <SourcePanel sources={message.sources} activePill={message.activePill} />
      )}
    </div>
  )
}
