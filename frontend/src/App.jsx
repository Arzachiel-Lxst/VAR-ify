import { useState } from 'react'
import { Upload, Play, AlertTriangle, CheckCircle, Hand, Flag, Loader2, Video, Download, Info, Clock, Zap } from 'lucide-react'

// API base URL from environment variable
const API_BASE = import.meta.env.VITE_API_URL || ''

function App() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [videoPreview, setVideoPreview] = useState(null)
  const [resultVideo, setResultVideo] = useState(null)

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile)
      setVideoPreview(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
      setResultVideo(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile)
      setVideoPreview(URL.createObjectURL(droppedFile))
      setResult(null)
      setError(null)
      setResultVideo(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('video', file)

    try {
      // Upload video
      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadRes.ok) throw new Error('Upload failed')

      const uploadData = await uploadRes.json()
      setUploading(false)
      setAnalyzing(true)

      // Analyze video
      const analyzeRes = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: uploadData.filename })
      })

      if (!analyzeRes.ok) throw new Error('Analysis failed')

      const resultData = await analyzeRes.json()
      console.log('API Response:', resultData)
      console.log('Video URL:', resultData.video_url)
      setResult(resultData)
      
      if (resultData.video_url) {
        setResultVideo(resultData.video_url)
        console.log('Result video set to:', resultData.video_url)
      } else {
        console.log('No video_url in response')
      }

    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
      setAnalyzing(false)
    }
  }

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3 rounded-lg mb-4">
            <Video className="w-8 h-8 text-white" />
            <h1 className="text-3xl font-bold text-white">VAR-ify</h1>
          </div>
          <h2 className="text-xl text-gray-300">Video Assistant Referee Analysis System</h2>
          <p className="text-gray-400 mt-2">Deteksi Handball & Offside Otomatis dengan AI</p>
        </div>

        {/* Info Section */}
        <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-4 mb-6">
          <div className="flex items-start gap-3">
            <Info className="w-6 h-6 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-blue-300 font-semibold mb-2">Apa itu VAR-ify?</h3>
              <p className="text-gray-300 text-sm mb-3">
                VAR-ify adalah sistem analisis video sepak bola berbasis AI yang dapat mendeteksi pelanggaran secara otomatis.
              </p>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div className="flex items-center gap-2 text-orange-300">
                  <Hand className="w-4 h-4" />
                  <span><strong>Handball:</strong> Deteksi sentuhan tangan dengan bola</span>
                </div>
                <div className="flex items-center gap-2 text-red-300">
                  <Flag className="w-4 h-4" />
                  <span><strong>Offside:</strong> Deteksi posisi offside pemain</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Warning Section */}
        <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-xl p-4 mb-6">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-yellow-400" />
            <div className="flex items-center gap-2">
              <span className="text-yellow-300 font-semibold">Durasi Maksimal: 15 detik</span>
              <span className="text-gray-400 text-sm">• Video lebih panjang akan dipotong otomatis</span>
            </div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Upload Section */}
          <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5" />
              Upload Video
            </h3>

            <div
              className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById('fileInput').click()}
            >
              <input
                id="fileInput"
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              {videoPreview ? (
                <div>
                  <video
                    src={videoPreview}
                    className="max-h-48 mx-auto rounded-lg mb-3"
                    controls
                  />
                  <p className="text-gray-300">{file?.name}</p>
                  <p className="text-gray-500 text-sm">
                    {(file?.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div>
                  <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                  <p className="text-gray-400">Drag & drop video atau klik untuk pilih</p>
                  <p className="text-gray-500 text-sm mt-1">MP4, MOV, AVI</p>
                </div>
              )}
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!file || uploading || analyzing}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : analyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Analyzing VAR...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Analyze Video
                </>
              )}
            </button>

            {error && (
              <div className="mt-4 bg-red-900/50 border border-red-500 rounded-lg p-3 text-red-300">
                {error}
              </div>
            )}
          </div>

          {/* Result Section */}
          <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              VAR Result
            </h3>

            {result ? (
              <div className="space-y-4">
                {/* Summary */}
                <div className={`p-4 rounded-lg ${
                  result.summary?.total_violations > 0 
                    ? 'bg-red-900/30 border border-red-500' 
                    : 'bg-green-900/30 border border-green-500'
                }`}>
                  {result.summary?.total_violations > 0 ? (
                    <div className="flex items-center gap-2 text-red-400">
                      <AlertTriangle className="w-6 h-6" />
                      <span className="text-xl font-bold">
                        {result.summary.total_violations} VIOLATION(S) DETECTED
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-green-400">
                      <CheckCircle className="w-6 h-6" />
                      <span className="text-xl font-bold">NO VIOLATIONS</span>
                    </div>
                  )}
                </div>

                {/* Handball */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-orange-400 mb-2">
                    <Hand className="w-5 h-5" />
                    <span className="font-semibold">HANDBALL</span>
                    <span className="ml-auto bg-orange-600 px-2 py-0.5 rounded text-white text-sm">
                      {result.handball?.length || 0}
                    </span>
                  </div>
                  {result.handball?.length > 0 ? (
                    <div className="space-y-2">
                      {result.handball.map((h, i) => (
                        <div key={i} className="text-gray-300 text-sm bg-gray-800 rounded p-2">
                          #{i + 1} | {h.timestamp?.toFixed(2)}s | {(h.confidence * 100).toFixed(0)}%
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">No handball detected</p>
                  )}
                </div>

                {/* Offside */}
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-red-400 mb-2">
                    <Flag className="w-5 h-5" />
                    <span className="font-semibold">OFFSIDE</span>
                    <span className="ml-auto bg-red-600 px-2 py-0.5 rounded text-white text-sm">
                      {result.offside?.length || 0}
                    </span>
                  </div>
                  {result.offside?.length > 0 ? (
                    <div className="space-y-2">
                      {result.offside.map((o, i) => (
                        <div key={i} className="text-gray-300 text-sm bg-gray-800 rounded p-2">
                          #{i + 1} | {o.timestamp?.toFixed(2)}s | {(o.confidence * 100).toFixed(0)}%
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-gray-500 text-sm">No offside detected</p>
                  )}
                </div>

                {/* Result Video */}
                {resultVideo && (
                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-white font-semibold">VAR Video</h4>
                      <button
                        onClick={() => {
                          const link = document.createElement('a')
                          link.href = `${API_BASE}${resultVideo}`
                          link.download = 'VAR_Result.mp4'
                          document.body.appendChild(link)
                          link.click()
                          document.body.removeChild(link)
                        }}
                        className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded-lg text-sm transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </div>
                    <video
                      src={`${API_BASE}${resultVideo}`}
                      className="w-full rounded-lg"
                      controls
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Video className="w-16 h-16 mx-auto mb-3 opacity-30" />
                <p>Upload dan analyze video untuk melihat hasil VAR</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          VAR-ify - Deteksi Handball & Offside menggunakan AI
        </div>
      </div>
    </div>
  )
}

export default App
