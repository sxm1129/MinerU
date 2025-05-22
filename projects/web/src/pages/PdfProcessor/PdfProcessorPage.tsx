import React, { useState } from 'react';
import './PdfProcessorPage.css'; // We'll create this CSS file later

const API_BASE_URL = '/api'; // Assuming Vite proxy is configured

interface ApiResponse {
  downloaded_pdf_path?: string;
  markdown_file_path?: string;
  md_content?: string;
  message?: string; // For success messages
}

const PdfProcessorPage: React.FC = () => {
  const [pdfUrl, setPdfUrl] = useState<string>('');
  const [apiResponse, setApiResponse] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!pdfUrl) {
      setError('Please enter a PDF URL.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setApiResponse(null);

    try {
      const response = await fetch(`${API_BASE_URL}/parse_pdf_from_url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pdf_url: pdfUrl }),
      });

      const data: ApiResponse & { error?: string } = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `API Error: ${response.statusText}`);
      }

      setApiResponse(data);
      // Clear URL input after successful submission if desired
      // setPdfUrl(''); 
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred.');
      }
      console.error('API call failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Helper to extract filename for PDF preview URL
  const getPdfPreviewFilename = (downloadPath?: string): string | null => {
    if (!downloadPath) return null;
    // Example path: "downloaded_pdfs/filename.pdf"
    // We need "filename.pdf" for the /get_pdf/ endpoint
    const parts = downloadPath.split('/');
    return parts.pop() || null;
  };

  const pdfPreviewFilename = getPdfPreviewFilename(apiResponse?.downloaded_pdf_path);

  return (
    <div className="pdf-processor-page">
      <h1>PDF Processor</h1>
      <form onSubmit={handleSubmit} className="url-form">
        <input
          type="url"
          value={pdfUrl}
          onChange={(e) => setPdfUrl(e.target.value)}
          placeholder="Enter PDF URL"
          required
          className="url-input"
        />
        <button type="submit" disabled={isLoading} className="submit-button">
          {isLoading ? 'Processing...' : 'Process PDF'}
        </button>
      </form>

      {error && <p className="error-message">Error: {error}</p>}
      
      {apiResponse?.message && <p className="success-message">{apiResponse.message}</p>}

      <div className="results-container">
        {pdfPreviewFilename && (
          <div className="pdf-preview-container">
            <h2>PDF Preview</h2>
            <iframe
              src={`${API_BASE_URL}/get_pdf/${pdfPreviewFilename}`}
              width="100%"
              height="600px"
              title="PDF Preview"
              className="pdf-iframe"
            ></iframe>
          </div>
        )}

        {apiResponse?.md_content && (
          <div className="markdown-container">
            <h2>Markdown Content</h2>
            {apiResponse.markdown_file_path && (
                <p>Markdown saved at: <code>{apiResponse.markdown_file_path}</code></p>
            )}
            <pre className="markdown-content">{apiResponse.md_content}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default PdfProcessorPage;
