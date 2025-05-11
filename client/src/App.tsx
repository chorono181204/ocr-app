import React, { useState } from 'react';
import { Box, Button, Typography, Paper, Stack } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [ocrResult, setOcrResult] = useState<string[] | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setSuccess(false);
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Only JPG or PNG files are accepted.');
      setSelectedFile(null);
      setPreview(null);
      return;
    }
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setError(null);
    setSuccess(false);
    setOcrResult(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await fetch('http://localhost:8000/ocr', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Upload thất bại');
      const data = await response.json();
      setOcrResult(data.text || []);
      setSuccess(true);
    } catch (err) {
      setError('Error uploading image.');
    } finally {
      setUploading(false);
    }
  };

  const handleCopy = () => {
    if (ocrResult) {
      navigator.clipboard.writeText(ocrResult.join('\n'));
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 1500);
    }
  };

  const handleDownload = () => {
    if (ocrResult) {
      const blob = new Blob([ocrResult.join('\n')], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (selectedFile?.name || 'result') + '.txt';
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <Box minHeight="100vh" width="100vw" bgcolor="#f5f6fa">
      {/* Header */}
      <Box width="100%" py={5} sx={{
        background: 'linear-gradient(90deg, #4f8cff 0%, #8f5cff 100%)',
        boxShadow: '0 4px 24px 0 rgba(79,140,255,0.10)',
        mb: 4
      }}>
        <Typography
          variant="h3"
          fontWeight={700}
          align="center"
          sx={{ color: '#fff', textShadow: '0 2px 8px rgba(0,0,0,0.15)', letterSpacing: 2 }}
        >
          Vietnamese Handwriting Recognition
        </Typography>
      </Box>
      {/* Upload Box */}
      <Box display="flex" justifyContent="center" alignItems="flex-start" minHeight="60vh">
        <Paper elevation={3} sx={{
          p: 4,
          width: '100%',
          maxWidth: 700,
          borderRadius: 4,
          border: '2px dashed #bdbdbd',
          background: '#fff',
          boxShadow: '0 8px 32px 0 rgba(79,140,255,0.08)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}>
          <Box display="flex" flexDirection="column" alignItems="center" mb={2}>
            <Box
              sx={{
                width: 80,
                height: 80,
                background: '#f0f4ff',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2
              }}
            >
              <svg width="64" height="48" viewBox="0 0 64 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="8" y="8" width="48" height="32" rx="4" fill="#F5F7FF" stroke="#E3E6F0" strokeWidth="2"/>
                <rect x="14" y="14" width="36" height="20" rx="2" fill="#E3E6F0"/>
                <rect x="20" y="18" width="24" height="12" rx="2" fill="#F5F7FF"/>
                <circle cx="28" cy="24" r="3" fill="#AEE571"/>
                <path d="M20 30L28 22L36 30H44" stroke="#AEE571" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Box>
            <Typography variant="h5" fontWeight={600} align="center" mb={1}>
              Image to Text Converter
            </Typography>
            <Typography variant="body1" color="text.secondary" align="center" mb={2}>
              An online image to text converter to extract text from images.
            </Typography>
          </Box>
          {/* Nếu đã chọn file thì luôn hiển thị thông tin file, preview, và nút Convert */}
          {selectedFile ? (
            <Box width="100%" mb={2}>
              <Typography variant="h6" mb={1}>Result</Typography>
              <Box position="relative" bgcolor="#fff" borderRadius={2} p={2} boxShadow={1}>
                {/* Nút Copy ở góc phải */}
                {ocrResult && (
                  <Button
                    size="small"
                    variant="text"
                    onClick={handleCopy}
                    sx={{ position: 'absolute', top: 8, right: 8, minWidth: 0, padding: 1, border: 'none', boxShadow: 'none' }}
                  >
                    <ContentCopyIcon color={copySuccess ? 'primary' : 'action'} />
                  </Button>
                )}
                <Box display="flex" alignItems="flex-start">
                  <img src={preview!} alt="preview" style={{ width: 60, height: 60, objectFit: 'cover', borderRadius: 8, marginRight: 16 }} />
                  <Box flex={1} minWidth={0}>
                    {uploading && (
                      <Box mt={1} width="100%">
                        <Box height={8} width="100%" bgcolor="#ede7f6" borderRadius={4} overflow="hidden">
                          <Box width="70%" height="100%" bgcolor="#b39ddb" sx={{ animation: 'progress 1.2s linear infinite alternate' }} />
                        </Box>
                        <style>{`@keyframes progress { 0%{width:10%} 100%{width:90%} }`}</style>
                      </Box>
                    )}
                    {/* Render từng dòng kết quả */}
                    {ocrResult && (
                      <Box mt={1}>
                        {ocrResult.map((line, idx) => (
                          <Typography key={idx} variant="body1" sx={{ whiteSpace: 'pre-line', wordBreak: 'break-word' }}>{line}</Typography>
                        ))}
                      </Box>
                    )}
                  </Box>
                </Box>
                {error && <Typography color="error" mt={2}>{error}</Typography>}
                {success && <Typography color="primary" mt={2}>Convert successful!</Typography>}
                {/* Nút quay về bước upload ảnh */}
                {(ocrResult || uploading) && (
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => { setSelectedFile(null); setPreview(null); setError(null); setSuccess(false); setOcrResult(null); setUploading(false); }}
                    sx={{ position: 'absolute', bottom: 8, right: 8, zIndex: 2 }}
                  >
                    Upload another image
                  </Button>
                )}
              </Box>
              {/* Nút Convert và Clear All chỉ hiện khi chưa có kết quả và không đang upload */}
              {(!uploading && !ocrResult) && (
                <Box display="flex" gap={2} mt={2}>
                  <Button variant="outlined" color="secondary" onClick={() => { setSelectedFile(null); setPreview(null); setError(null); setSuccess(false); setOcrResult(null); }}>
                    Clear All
                  </Button>
                  <Button
                    variant="contained"
                    color="success"
                    onClick={handleUpload}
                    disabled={!selectedFile || uploading}
                  >
                    Convert
                  </Button>
                </Box>
              )}
            </Box>
          ) : null}
          {/* Ẩn phần chọn file khi đã chọn file */}
          {!selectedFile && (
            <>
              <Button
                variant="contained"
                component="label"
                sx={{
                  minWidth: 180,
                  fontWeight: 600,
                  fontSize: 16,
                  py: 1.5,
                  mb: 2
                }}
              >
                Browse
                <input
                  type="file"
                  accept="image/jpeg,image/png"
                  hidden
                  onChange={handleFileChange}
                />
              </Button>
              <Typography variant="body2" color="text.secondary" mb={2}>
                Supported formats: JPG, PNG
              </Typography>
            </>
          )}
        </Paper>
      </Box>
    </Box>
  );
}

export default App;
