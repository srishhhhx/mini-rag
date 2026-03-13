'use client'

import { useChat } from '@/hooks/useChat'
import FileUpload from '@/components/FileUpload'
import UploadProgress from '@/components/UploadProgress'
import ChatWindow from '@/components/ChatWindow'
import ChatInput from '@/components/ChatInput'

export default function Home() {
  const {
    state,
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
  } = useChat()

  /* ── Upload Screen ─────────────────────────────────────────────── */
  if (state === 'upload') {
    return (
      <main className="upload-screen">
        <div>
          <h1 className="upload-screen__title">PDF Chat</h1>
          <p className="upload-screen__subtitle">
            Ask questions · Get answers · Grounded in your document
          </p>
        </div>

        <div className="upload-card">
          <div className="upload-card__header">
            <h2 className="upload-card__title">Drop</h2>
            <p className="upload-card__types">PDF · max 20MB</p>
          </div>

          <FileUpload
            onUpload={handleUpload}
            isUploading={isUploading}
            error={uploadError}
          />
        </div>
      </main>
    )
  }

  /* ── Processing Screen ─────────────────────────────────────────── */
  if (state === 'processing') {
    return (
      <UploadProgress
        filename={file?.name ?? 'document.pdf'}
        error={processingError}
        onRetry={resetSession}
      />
    )
  }

  /* ── Chat Screen ───────────────────────────────────────────────── */
  return (
    <main className="chat-screen">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header__info">
          <h1 className="chat-header__title" title={docTitle ?? ''}>
            {docTitle ?? file?.name ?? 'Document'}
          </h1>
          {docTopic && (
            <p className="chat-header__subtitle">{docTopic}</p>
          )}
        </div>
        <button
          className="btn-new-doc"
          onClick={resetSession}
          id="new-doc-btn"
          aria-label="Upload a new document"
        >
          ↩ New document
        </button>
      </header>

      {/* Messages */}
      <ChatWindow
        messages={messages}
        onToggleSources={toggleSources}
        onPillClick={togglePill}
      />

      {/* Input */}
      <ChatInput onSend={sendMessage} disabled={isStreaming} />
    </main>
  )
}
