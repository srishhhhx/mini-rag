'use client'

import { KeyboardEvent, useRef, useState } from 'react'

interface ChatInputProps {
  onSend: (question: string) => void
  disabled: boolean
}

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const q = value.trim()
    if (!q || disabled) return
    onSend(q)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  return (
    <div className="chat-input-bar">
      <form
        className="chat-input-form"
        onSubmit={(e) => { e.preventDefault(); submit() }}
      >
        <textarea
          ref={textareaRef}
          id="chat-input"
          className="chat-input"
          value={value}
          onChange={(e) => { setValue(e.target.value); autoResize() }}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your document…"
          disabled={disabled}
          rows={1}
          aria-label="Chat message input"
        />
        <button
          type="submit"
          className="btn-send"
          disabled={disabled || !value.trim()}
          aria-label="Send message"
          id="send-btn"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
      <p className="chat-input-bar__hint">Enter to send · Shift+Enter for new line</p>
    </div>
  )
}
