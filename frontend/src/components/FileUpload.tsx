'use client'

import { useRef, useState, DragEvent, ChangeEvent } from 'react'

interface FileUploadProps {
  onUpload: (file: File) => void
  isUploading: boolean
  error: string | null
}

export default function FileUpload({ onUpload, isUploading, error }: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(f: File) {
    if (!f.name.toLowerCase().endsWith('.pdf')) {
      return
    }
    setSelectedFile(f)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  function handleSubmit() {
    if (selectedFile) onUpload(selectedFile)
  }

  return (
    <>
      <div
        className={`dropzone${isDragging ? ' dragover' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !isUploading && inputRef.current?.click()}
        role="button"
        aria-label="Drop PDF here or click to browse"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <div className="dropzone__icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        {selectedFile ? (
          <div className="dropzone__file-selected">
            📄 {selectedFile.name}
          </div>
        ) : (
          <>
            <p className="dropzone__label">
              Drop file or <strong>browse</strong>
            </p>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>PDF only · max 20MB</p>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleChange}
          style={{ display: 'none' }}
          aria-hidden
        />
      </div>

      {error && <div className="upload-error">{error}</div>}

      <button
        className="btn-upload"
        onClick={handleSubmit}
        disabled={!selectedFile || isUploading}
        aria-label="Upload PDF"
        id="upload-btn"
      >
        {isUploading ? (
          <>
            <span className="spinner" style={{ width: 14, height: 14, borderWidth: 1.5 }} />
            Uploading…
          </>
        ) : (
          <>
            Upload
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          </>
        )}
      </button>
    </>
  )
}
