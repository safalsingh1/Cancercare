/**
 * CancerCare React Frontend Application
 *
 * Core State Layout:
 * - activeTab tracks the display of the Classifier, Genomic Explorer, and Chatbot
 * - Classifier handlers process image uploading, preview logic, and formData prediction fetching.
 * - Genomic handlers submit JSON-encoded gene or variant queries to the backend.
 * - Chatbot handlers maintain message history and fetch string answers from the Gemini AI route.
 */
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('classifier');

  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [queryType, setQueryType] = useState('gene');
  const [query, setQuery] = useState('');
  const [genomeResult, setGenomeResult] = useState(null);
  const [genomeLoading, setGenomeLoading] = useState(false);
  const [genomeError, setGenomeError] = useState(null);

  const [chatMessages, setChatMessages] = useState([
    { text: "Hello! I am an emotional support AI. How can I help you today?", isBot: true }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => { scrollToBottom(); }, [chatMessages]);

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageSelect(file);
    else setError('Please upload a valid image file.');
  };
  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) handleImageSelect(file);
  };
  const handleImageSelect = (file) => {
    setError(null); setPrediction(null); setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };
  const handlePredict = async () => {
    if (!selectedImage) return;
    setLoading(true); setError(null);
    const formData = new FormData();
    formData.append('file', selectedImage);
    try {
      const response = await fetch('http://localhost:8000/predict', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Prediction failed. Please ensure the backend is running.');
      const data = await response.json();
      setPrediction(data);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  const handleGenomeSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setGenomeLoading(true); setGenomeError(null); setGenomeResult(null);
    try {
      const response = await fetch('http://localhost:8000/genomic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_type: queryType, query: query.trim() })
      });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Failed to fetch genomic data');
      setGenomeResult(data);
    } catch (err) { setGenomeError(err.message); } finally { setGenomeLoading(false); }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const userMsg = chatInput.trim();
    setChatMessages(prev => [...prev, { text: userMsg, isBot: false }]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: userMsg })
      });
      const data = await response.json();
      setChatMessages(prev => [...prev, { text: data.response || "Server error", isBot: true }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { text: "Error: Could not reach chat server.", isBot: true }]);
    } finally { setChatLoading(false); }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>CancerCare Assistant</h1>
        <p>A comprehensive platform for classification, genomic explorer, and emotional support.</p>

        <div className="tabs">
          <button className={`tab ${activeTab === 'classifier' ? 'active' : ''}`} onClick={() => setActiveTab('classifier')}>Image Classifier</button>
          <button className={`tab ${activeTab === 'genome' ? 'active' : ''}`} onClick={() => setActiveTab('genome')}>Genomic Explorer</button>
          <button className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>AI Support Chat</button>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'classifier' && (
          <div className="tab-pane fade-in">
            <div className="upload-section">
              <div
                className={`dropzone ${preview ? 'has-image' : ''}`}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => { if (!preview) document.getElementById('file-input').click(); }}
              >
                {preview ? (
                  <div className="preview-container">
                    <img src={preview} alt="Preview" className="image-preview" />
                    <button className="change-btn" onClick={(e) => {
                      e.stopPropagation(); setPreview(null); setSelectedImage(null); setPrediction(null);
                    }}>Change Image</button>
                  </div>
                ) : (
                  <div className="dropzone-text">
                    <i className="upload-icon">📁</i>
                    <p>Drag & Drop your slide image here</p>
                    <span>or click to browse</span>
                  </div>
                )}
                <input type="file" id="file-input" accept="image/*" onChange={handleFileInput} style={{ display: 'none' }} />
              </div>

              <button className="primary-btn" onClick={handlePredict} disabled={!selectedImage || loading}>
                {loading ? <span className="spinner"></span> : 'Analyze Image'}
              </button>
              {error && <div className="error-message">{error}</div>}
            </div>

            {prediction && (
              <div className="result-section fade-in">
                <h2>Analysis Result</h2>
                <div className="result-card">
                  <div className="result-item">
                    <span className="label">Subtype Identified:</span>
                    <span className="value highlight">{prediction.prediction}</span>
                  </div>
                  <div className="result-item">
                    <span className="label">Confidence Score:</span>
                    <span className="value">{prediction.confidence.toFixed(2)}%</span>
                  </div>
                  <div className="confidence-bar-container">
                    <div className="confidence-bar" style={{ width: `${prediction.confidence}%` }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'genome' && (
          <div className="tab-pane fade-in">
            <h2 className="section-title">Genomic Data Explorer</h2>
            <form onSubmit={handleGenomeSubmit} className="genome-form">
              <div className="form-group">
                <label>Select Query Type:</label>
                <select value={queryType} onChange={(e) => setQueryType(e.target.value)}>
                  <option value="gene">Gene</option>
                  <option value="variant">Variant (rsID)</option>
                </select>
              </div>
              <div className="form-group">
                <label>{queryType === 'gene' ? 'Enter Gene Symbol (e.g., BRCA1):' : 'Enter Variant rsID (e.g., rs121913529):'}</label>
                <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} required placeholder={queryType === 'gene' ? 'BRCA1' : 'rs121913529'} />
              </div>
              <button type="submit" className="primary-btn" disabled={genomeLoading || !query.trim()}>
                {genomeLoading ? <span className="spinner"></span> : 'Search Database'}
              </button>
            </form>
            {genomeError && <div className="error-message mt-4">{genomeError}</div>}
            {genomeResult && (
              <div className="result-section fade-in">
                <h2>Result Data</h2>
                <pre className="json-result">{JSON.stringify(genomeResult, null, 2)}</pre>
              </div>
            )}
          </div>
        )}

        {activeTab === 'chat' && (
          <div className="tab-pane fade-in chat-container">
            <h2 className="section-title">Emotional Support Chatbot</h2>
            <div className="chat-window">
              {chatMessages.map((msg, idx) => (
                <div key={idx} className={`message-row ${msg.isBot ? 'bot-row' : 'user-row'}`}>
                  <div className={`message-bubble ${msg.isBot ? 'bot-bubble' : 'user-bubble'}`}>
                    {msg.text}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="message-row bot-row fade-in">
                  <div className="message-bubble bot-bubble typing-indicator">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <form onSubmit={handleChatSubmit} className="chat-input-form">
              <input type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)} placeholder="Type a message..." disabled={chatLoading} />
              <button type="submit" className="send-btn" disabled={chatLoading || !chatInput.trim()}>Send</button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
