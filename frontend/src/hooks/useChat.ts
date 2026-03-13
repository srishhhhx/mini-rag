'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { uploadPDF, getSessionStatus, deleteSession, createChatStream, SourceChunk } from '@/lib/api'

export type AppState = 'upload' | 'processing' | 'chat'

export interface Message {
  id: string
  role: 'user' | 'ai'
  content: string
  streaming: boolean
  sources: SourceChunk[]
  confident: boolean
  showSources: boolean
  activePill: number | null
}

function makeId() {
  return Math.random().toString(36).slice(2)
}

function getOrCreateSessionId(): string {
  if (typeof window === 'undefined') return makeId()
  let id = sessionStorage.getItem('pdf_session_id')
  if (!id) {
    id = crypto.randomUUID()
    sessionStorage.setItem('pdf_session_id', id)
  }
  return id
}

export function useChat() {
  const [state, setState] = useState<AppState>('upload')
  const [sessionId] = useState<string>(getOrCreateSessionId)
  const [file, setFile] = useState<File | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [docTitle, setDocTitle] = useState<string | null>(null)
  const [docTopic, setDocTopic] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [processingError, setProcessingError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const abortRef = useRef<(() => void) | null>(null)

  // ── Upload ──────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (f: File) => {
    setUploadError(null)
    setIsUploading(true)
    try {
      await uploadPDF(f, sessionId)
      setFile(f)
      setState('processing')
      startPolling(sessionId)
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }, [sessionId])

  // ── Poll ingestion status ────────────────────────────────────────────
  const startPolling = useCallback((sid: string) => {
    setProcessingError(null)
    pollRef.current = setInterval(async () => {
      try {
        const status = await getSessionStatus(sid)
        if (status.status === 'ready') {
          clearInterval(pollRef.current!)
          setDocTitle(status.doc_title)
          setDocTopic(status.doc_topic)
          setMessages([])
          setState('chat')
        } else if (status.status === 'error') {
          clearInterval(pollRef.current!)
          setProcessingError(status.error_message || 'Ingestion failed')
        }
      } catch {
        // network blip — keep polling
      }
    }, 1500)
  }, [])

  // ── Send message ─────────────────────────────────────────────────────
  const sendMessage = useCallback((question: string) => {
    if (!question.trim() || isStreaming) return

    const userMsg: Message = {
      id: makeId(),
      role: 'user',
      content: question.trim(),
      streaming: false,
      sources: [],
      confident: true,
      showSources: false,
      activePill: null,
    }
    const aiMsgId = makeId()
    const aiMsg: Message = {
      id: aiMsgId,
      role: 'ai',
      content: '',
      streaming: true,
      sources: [],
      confident: true,
      showSources: false,
      activePill: null,
    }

    setMessages(prev => [...prev, userMsg, aiMsg])
    setIsStreaming(true)

    abortRef.current = createChatStream(
      sessionId,
      question.trim(),
      (token) => {
        setMessages(prev =>
          prev.map(m => m.id === aiMsgId ? { ...m, content: m.content + token } : m)
        )
      },
      (sources, confident) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === aiMsgId ? { ...m, streaming: false, sources, confident } : m
          )
        )
        setIsStreaming(false)
      },
      (err) => {
        setMessages(prev =>
          prev.map(m =>
            m.id === aiMsgId
              ? { ...m, content: `Error: ${err}`, streaming: false }
              : m
          )
        )
        setIsStreaming(false)
      }
    )
  }, [sessionId, isStreaming])

  // ── Toggle source panel ───────────────────────────────────────────────
  const toggleSources = useCallback((msgId: string) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === msgId
          ? { ...m, showSources: !m.showSources, activePill: null }
          : m
      )
    )
  }, [])

  // ── Toggle active pill ────────────────────────────────────────────────
  const togglePill = useCallback((msgId: string, idx: number) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === msgId
          ? {
              ...m,
              showSources: true,
              activePill: m.activePill === idx ? null : idx,
            }
          : m
      )
    )
  }, [])

  // ── New document ──────────────────────────────────────────────────────
  const resetSession = useCallback(async () => {
    abortRef.current?.()
    clearInterval(pollRef.current!)
    try { await deleteSession(sessionId) } catch { /* ignore */ }
    sessionStorage.removeItem('pdf_session_id')
    // Generate fresh session ID by reloading
    window.location.reload()
  }, [sessionId])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.()
      clearInterval(pollRef.current!)
    }
  }, [])

  return {
    state,
    sessionId,
    file,
    messages,
    isStreaming,
    docTitle,
    docTopic,
    uploadError,
    processingError,
    isUploading,
    handleUpload,
    sendMessage,
    toggleSources,
    togglePill,
    resetSession,
  }
}
