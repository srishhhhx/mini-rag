'use client'

import { useEffect, useRef } from 'react'
import { Message } from '@/hooks/useChat'
import MessageBubble from './MessageBubble'

interface ChatWindowProps {
  messages: Message[]
  onToggleSources: (id: string) => void
  onPillClick: (id: string, idx: number) => void
}

export default function ChatWindow({ messages, onToggleSources, onPillClick }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll on new message or token
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="chat-messages">
        <div className="chat-empty">
          <div className="chat-empty__icon"></div>
          <p className="chat-empty__label">Ask anything about the document</p>
        </div>
      </div>
    )
  }

  return (
    <div className="chat-messages" id="chat-messages">
      {messages.map(msg => (
        <MessageBubble
          key={msg.id}
          message={msg}
          onToggleSources={() => onToggleSources(msg.id)}
          onPillClick={(idx) => onPillClick(msg.id, idx)}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
