import { useState } from 'react'

function App() {
  const [prompt, setPrompt] = useState('')
  const [image, setImage] = useState<string | null>(null)

  const [reference, setReference] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      alert('Please enter a prompt')
      return
    }

    setLoading(true)
    setError(null)
    setImage(null)
    setReference(null)

    try {
      const response = await fetch(
        import.meta.env.VITE_BACKEND_URL, 
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ prompt: prompt }),
        }
      )

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.error) {
        throw new Error(data.error)
      }

      if (data.image) {
        setImage(data.image)
      }

      if (data.reference_image) {
        setReference(data.reference_image)
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred'
      console.error('Error generating image:', err)
      setError(errorMessage)
      alert(`Failed to generate image: ${errorMessage}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        
        <div className="text-center mb-10">
          <h1 className="text-5xl font-extrabold mb-4 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 bg-clip-text text-transparent tracking-tight">
            Cyberpunk RAG Image Generator
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto leading-relaxed">
            This system uses <span className="text-purple-400 font-semibold">Retrieval-Augmented Generation (RAG)</span>. 
            It searches a curated library of <span className="text-pink-400 font-semibold">Cyberpunk artwork </span> 
            to find the perfect style match before generating your image.
          </p>
        </div>

        <div className="mb-8">
          <label htmlFor="prompt" className="block text-sm font-medium mb-2 text-gray-300 ml-1">
            Describe your scene
          </label>
          <div className="flex flex-col sm:flex-row gap-3">
            <input
              id="prompt"
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !loading) {
                  handleGenerate()
                }
              }}
              placeholder="e.g., A futuristic detective in the rain..."
              className="flex-1 px-5 py-4 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-100 placeholder-gray-500 text-lg transition-all"
              disabled={loading}
            />
            <button
              onClick={handleGenerate}
              disabled={loading || !prompt.trim()}
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-xl font-bold text-lg shadow-lg hover:shadow-purple-500/25 transition-all duration-200 whitespace-nowrap"
            >
              {loading ? 'Generating...' : 'Generate Art'}
            </button>
          </div>
        </div>

        {loading && (
          <div className="flex flex-col items-center justify-center py-16 bg-gray-800/30 rounded-2xl border border-gray-800 border-dashed">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mb-4"></div>
            <p className="text-gray-400 animate-pulse">Searching vector database & diffusing pixels...</p>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-xl text-red-200 text-center">
            <p className="font-semibold">Generation Failed</p>
            <p className="text-sm opacity-80">{error}</p>
          </div>
        )}

        {image && (
          <div className="space-y-6 animate-fade-in">
            {/* Main Generated Image */}
            <div className="bg-gray-800 rounded-2xl p-2 border border-gray-700 shadow-2xl">
              <img
                src={image}
                alt="Generated"
                className="w-full h-auto rounded-xl"
              />
            </div>
            
            {/* --- UPDATED STYLE REFERENCE SECTION --- */}
            {reference && (
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 flex items-center gap-4">
                <div className="bg-purple-500/10 p-2 rounded-lg shrink-0">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div className="flex-1">
                   <p className="text-sm font-semibold text-purple-400 uppercase tracking-wider mb-2">RAG Style Match Used</p>
                   {/* Replaced text with a small image thumbnail */}
                   <img 
                    src={reference} 
                    alt="Style Reference Thumbnail" 
                    className="w-32 h-32 object-cover rounded-lg border-2 border-purple-500/30 shadow-sm hover:scale-105 transition-transform"
                   />
                </div>
              </div>
            )}
            {/* --------------------------------------- */}

          </div>
        )}
      </div>
    </div>
  )
}

export default App