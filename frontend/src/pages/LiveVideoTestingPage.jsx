import React, { useState, useEffect, useRef } from 'react';
import { getCameras } from '../services/gisService';

const STAGES = [
  { id: 'VIDEO_LOADING', label: '1. Video Loaded', icon: 'file_open' },
  { id: 'MODEL_INITIALIZATION', label: '2. Camera & Pipeline Bound', icon: 'memory' },
  { id: 'AI_PERCEPTION', label: '3. Vehicle & Plate Detection', icon: 'radar' },
  { id: 'BYTETRACK_TRACKING', label: '4. ByteTrack Motion Tracking', icon: 'timeline' },
  { id: 'PADDLE_OCR', label: '5. PaddleOCR License Plates', icon: 'document_scanner' },
  { id: 'REID_EMBEDDING', label: '6. OSNet-AIN Vehicle Re-ID', icon: 'fingerprint' },
  { id: 'DATABASE_INGESTION', label: '7. Database Ingestion (PostGIS)', icon: 'database' }
];

export default function LiveVideoTestingPage({ setRoute }) {
  const [cameras, setCameras] = useState([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileValidationErr, setFileValidationErr] = useState(null);
  const [useSampleVideo, setUseSampleVideo] = useState(false);

  // Job execution states: IDLE | UPLOADING | QUEUED | PROCESSING | COMPLETED | FAILED
  const [jobState, setJobState] = useState({
    status: 'IDLE',
    stage: 'IDLE',
    stage_label: '',
    progress_percent: 0,
    job_id: null,
    metrics: null,
    video_url: null,
    error: null,
  });

  const [uploadProgress, setUploadProgress] = useState(0);
  const pollIntervalRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load real cameras from FastAPI backend
  useEffect(() => {
    getCameras()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setCameras(data);
          setSelectedCameraId(data[0].id);
        }
      })
      .catch((err) => {
        console.warn('Failed to load cameras for testing:', err);
      });
  }, []);

  // Poll job status while active
  useEffect(() => {
    if (jobState.status === 'QUEUED' || jobState.status === 'PROCESSING') {
      pollIntervalRef.current = setInterval(async () => {
        if (!jobState.job_id) return;
        try {
          const res = await fetch(`/api/v1/video-testing/jobs/${jobState.job_id}`);
          if (res.ok) {
            const data = await res.json();
            setJobState((prev) => ({
              ...prev,
              status: data.status,
              stage: data.stage,
              stage_label: data.stage_label || '',
              progress_percent: data.progress_percent || 0,
              metrics: data.metrics || prev.metrics,
              video_url: data.video_relative_url || prev.video_url,
              error: data.error || null,
            }));

            if (data.status === 'COMPLETED' || data.status === 'FAILED') {
              clearInterval(pollIntervalRef.current);
            }
          }
        } catch (e) {
          console.warn('Polling error:', e);
        }
      }, 2000);
    } else {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    }

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [jobState.status, jobState.job_id]);

  const selectedCamera = cameras.find((c) => c.id === selectedCameraId) || null;

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    setFileValidationErr(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }

    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
    const nameLower = file.name.toLowerCase();
    const hasValidExt = validExtensions.some((ext) => nameLower.endsWith(ext));

    if (!hasValidExt) {
      setFileValidationErr(`Unsupported file type. Please upload MP4, AVI, MOV, or MKV.`);
      setSelectedFile(null);
      return;
    }

    const maxSizeMb = 150;
    if (file.size > maxSizeMb * 1024 * 1024) {
      setFileValidationErr(`File exceeds ${maxSizeMb} MB limit (${(file.size / 1024 / 1024).toFixed(1)} MB).`);
      setSelectedFile(null);
      return;
    }

    setUseSampleVideo(false);
    setSelectedFile(file);
  };

  const handleLoadSampleVideo = async () => {
    try {
      setFileValidationErr(null);
      // Fetch sample corridor video from backend
      const sampleBlob = await fetch('/api/v1/video-testing/jobs/test-smoke-001/video')
        .then((r) => r.ok ? r.blob() : null)
        .catch(() => null);

      if (sampleBlob) {
        const file = new File([sampleBlob], 'test_corridor_sample.mp4', { type: 'video/mp4' });
        setSelectedFile(file);
        setUseSampleVideo(true);
      } else {
        setFileValidationErr('Sample corridor test video is not currently available from the server. Please upload an MP4 or AVI file directly.');
        setSelectedFile(null);
        setUseSampleVideo(false);
      }
    } catch (err) {
      console.error('Sample video load error:', err);
      setFileValidationErr('Failed to retrieve sample video from backend. Please upload a video file directly.');
      setSelectedFile(null);
      setUseSampleVideo(false);
    }
  };

  const handleStartProcessing = async () => {
    if (!selectedCameraId) {
      setFileValidationErr('Please select a valid surveillance camera node.');
      return;
    }
    if (!selectedFile) {
      setFileValidationErr('Please select a video file to test.');
      return;
    }

    setJobState({
      status: 'UPLOADING',
      stage: 'UPLOADING',
      stage_label: 'Uploading video payload to NETRA backend...',
      progress_percent: 5,
      job_id: null,
      metrics: null,
      video_url: null,
      error: null,
    });
    setUploadProgress(10);

    const formData = new FormData();
    formData.append('camera_id', selectedCameraId);
    formData.append('file', selectedFile);

    try {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/v1/video-testing/jobs', true);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = Math.round((event.loaded / event.total) * 100);
          setUploadProgress(percent);
        }
      };

      xhr.onload = () => {
        if (xhr.status === 201 || xhr.status === 200) {
          const res = JSON.parse(xhr.responseText);
          setJobState({
            status: 'QUEUED',
            stage: 'QUEUED',
            stage_label: 'Video uploaded. Initializing AI perception pipeline...',
            progress_percent: 15,
            job_id: res.job_id,
            metrics: null,
            video_url: null,
            error: null,
          });
        } else {
          let errDetail = 'Failed to upload video';
          try {
            const errRes = JSON.parse(xhr.responseText);
            errDetail = errRes.detail || errDetail;
          } catch (_) {}
          setJobState({
            status: 'FAILED',
            stage: 'FAILED',
            stage_label: 'Upload rejected by backend.',
            progress_percent: 100,
            job_id: null,
            metrics: null,
            video_url: null,
            error: errDetail,
          });
        }
      };

      xhr.onerror = () => {
        setJobState({
          status: 'FAILED',
          stage: 'FAILED',
          stage_label: 'Network error communicating with backend.',
          progress_percent: 100,
          job_id: null,
          metrics: null,
          video_url: null,
          error: 'Connection to NETRA backend failed. Ensure backend server is running.',
        });
      };

      xhr.send(formData);
    } catch (e) {
      setJobState({
        status: 'FAILED',
        stage: 'FAILED',
        stage_label: 'Failed to initiate upload.',
        progress_percent: 100,
        job_id: null,
        metrics: null,
        video_url: null,
        error: e.message || 'Unknown upload exception',
      });
    }
  };

  const handleReset = () => {
    setJobState({
      status: 'IDLE',
      stage: 'IDLE',
      stage_label: '',
      progress_percent: 0,
      job_id: null,
      metrics: null,
      video_url: null,
      error: null,
    });
    setSelectedFile(null);
    setUseSampleVideo(false);
    setFileValidationErr(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const isBusy = jobState.status === 'UPLOADING' || jobState.status === 'QUEUED' || jobState.status === 'PROCESSING';

  return (
    <div style={{ padding: '1.5rem 2rem', maxWidth: '1600px', margin: '0 auto' }}>
      {/* Page Title Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: '#00677d', letterSpacing: '0.06em' }}>
            Optical Intelligence Verification Unit
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: 800, color: '#001e40', margin: 0 }}>
            Live Video Testing Console
          </h1>
          <p style={{ fontSize: '12.5px', color: '#64748b', margin: '2px 0 0 0' }}>
            Upload raw surveillance feeds, associate with registered nodes, and execute the full NETRA AI Pipeline
          </p>
        </div>

        {/* Status Header Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            fontSize: '11.5px',
            fontWeight: 700,
            padding: '5px 12px',
            borderRadius: '6px',
            backgroundColor: isBusy ? 'rgba(234, 179, 8, 0.12)' : jobState.status === 'COMPLETED' ? 'rgba(22, 163, 74, 0.12)' : 'rgba(56, 189, 248, 0.1)',
            color: isBusy ? '#ca8a04' : jobState.status === 'COMPLETED' ? '#16a34a' : '#0284c7',
            border: `1px solid ${isBusy ? 'rgba(234, 179, 8, 0.3)' : jobState.status === 'COMPLETED' ? 'rgba(22, 163, 74, 0.3)' : 'rgba(56, 189, 248, 0.25)'}`,
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
              {isBusy ? 'sync' : jobState.status === 'COMPLETED' ? 'check_circle' : 'smart_display'}
            </span>
            <span>{jobState.status}</span>
          </div>

          {jobState.status === 'COMPLETED' && (
            <button
              onClick={handleReset}
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #cbd5e1',
                padding: '5px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: 600,
                color: '#334155',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>refresh</span>
              Test Another Video
            </button>
          )}
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 420px) 1fr', gap: '1.5rem', alignItems: 'start' }}>

        {/* ========================================================= */}
        {/* LEFT COLUMN: CAMERA SELECTOR + VIDEO UPLOAD CONTROLS      */}
        {/* ========================================================= */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          {/* Card 1: Camera Node Selection */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <span className="material-symbols-outlined" style={{ color: '#003366', fontSize: '20px' }}>videocam</span>
              <h2 style={{ fontSize: '14px', fontWeight: 800, color: '#001e40', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                1. Select Camera Node
              </h2>
            </div>
            <p style={{ fontSize: '11.5px', color: '#64748b', margin: '0 0 12px 0' }}>
              Associate the test video stream with a registered physical or corridor camera.
            </p>

            {cameras.length === 0 ? (
              <div style={{ padding: '12px', backgroundColor: '#f8fafc', borderRadius: '6px', fontSize: '12px', color: '#64748b', textAlign: 'center' }}>
                Loading verified cameras from database...
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <select
                  value={selectedCameraId}
                  onChange={(e) => setSelectedCameraId(e.target.value)}
                  disabled={isBusy}
                  style={{
                    width: '100%',
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid #cbd5e1',
                    fontSize: '12.5px',
                    fontWeight: 700,
                    color: '#001e40',
                    backgroundColor: isBusy ? '#f1f5f9' : '#ffffff',
                    outline: 'none',
                    cursor: isBusy ? 'not-allowed' : 'pointer'
                  }}
                >
                  {cameras.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} — {c.name || `Camera ${c.id}`} ({c.roadName || c.road_name || 'Highway Corridor'})
                    </option>
                  ))}
                </select>

                {/* Camera Telemetry Metadata Box */}
                {selectedCamera && (
                  <div style={{
                    padding: '10px 12px',
                    borderRadius: '6px',
                    backgroundColor: '#f8fafc',
                    border: '1px solid #e2e8f0',
                    fontSize: '11.5px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Camera Node:</span>
                      <span style={{ fontWeight: 700, color: '#001e40' }}>{selectedCamera.id}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Location:</span>
                      <span style={{ fontWeight: 600, color: '#1e293b' }}>{selectedCamera.locationName || selectedCamera.name || 'Coimbatore Grid'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Corridor Road:</span>
                      <span style={{ fontWeight: 600, color: '#1e293b' }}>{selectedCamera.roadName || selectedCamera.road_name || 'Avinashi Road'}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Status / FPS:</span>
                      <span style={{ fontWeight: 700, color: selectedCamera.status === 'online' ? '#16a34a' : '#d97706' }}>
                        {selectedCamera.status?.toUpperCase() || 'ONLINE'} • {selectedCamera.fps || 30} FPS
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#64748b' }}>Geo Coordinates:</span>
                      <span style={{ fontFamily: 'monospace', fontSize: '11px', color: '#475569' }}>
                        {selectedCamera.latitude?.toFixed(4)}, {selectedCamera.longitude?.toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Card 2: Video File Input */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <span className="material-symbols-outlined" style={{ color: '#003366', fontSize: '20px' }}>upload_file</span>
              <h2 style={{ fontSize: '14px', fontWeight: 800, color: '#001e40', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                2. Test Video Input
              </h2>
            </div>
            <p style={{ fontSize: '11.5px', color: '#64748b', margin: '0 0 12px 0' }}>
              Select a traffic video clip (MP4, AVI, MOV). Max size: 150 MB.
            </p>

            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              accept=".mp4,.avi,.mov,.mkv"
              onChange={handleFileChange}
              disabled={isBusy}
              style={{ display: 'none' }}
            />

            {/* Custom Drop / Upload Area */}
            <div
              onClick={() => !isBusy && fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${selectedFile ? '#003366' : '#cbd5e1'}`,
                borderRadius: '8px',
                padding: '1.5rem 1rem',
                textAlign: 'center',
                backgroundColor: selectedFile ? 'rgba(0, 51, 102, 0.03)' : '#f8fafc',
                cursor: isBusy ? 'not-allowed' : 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              <span className="material-symbols-outlined" style={{
                fontSize: '36px',
                color: selectedFile ? '#003366' : '#94a3b8',
                marginBottom: '6px'
              }}>
                {selectedFile ? 'video_file' : 'cloud_upload'}
              </span>

              {selectedFile ? (
                <div>
                  <div style={{ fontSize: '12.5px', fontWeight: 700, color: '#001e40', wordBreak: 'break-all' }}>
                    {selectedFile.name}
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '3px' }}>
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB • {selectedFile.type || 'video/mp4'}
                  </div>
                  <div style={{ fontSize: '10.5px', color: '#0284c7', marginTop: '6px', fontWeight: 600 }}>
                    Click to choose a different video
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#1e293b' }}>
                    Click to Choose Traffic Video
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                    Supports MP4, AVI, MOV (H.264 / AVC)
                  </div>
                </div>
              )}
            </div>

            {/* Sample Video Quick-Load Helper */}
            <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <button
                type="button"
                onClick={handleLoadSampleVideo}
                disabled={isBusy}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#00677d',
                  fontSize: '11.5px',
                  fontWeight: 600,
                  textDecoration: 'underline',
                  cursor: isBusy ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: 0
                }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>smart_display</span>
                Load System Sample Clip (test.mp4)
              </button>

              {useSampleVideo && (
                <span style={{ fontSize: '10.5px', fontWeight: 700, color: '#16a34a', backgroundColor: 'rgba(22, 163, 74, 0.1)', padding: '2px 6px', borderRadius: '4px' }}>
                  Sample Loaded
                </span>
              )}
            </div>

            {/* Validation Error Banner */}
            {fileValidationErr && (
              <div style={{
                marginTop: '10px',
                padding: '8px 12px',
                borderRadius: '6px',
                backgroundColor: 'rgba(220, 38, 38, 0.1)',
                border: '1px solid rgba(220, 38, 38, 0.25)',
                color: '#dc2626',
                fontSize: '11.5px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>error</span>
                <span>{fileValidationErr}</span>
              </div>
            )}

            {/* Execution Trigger Button */}
            <button
              onClick={handleStartProcessing}
              disabled={isBusy || !selectedFile || !selectedCameraId}
              style={{
                marginTop: '1.25rem',
                width: '100%',
                padding: '10px 16px',
                borderRadius: '6px',
                backgroundColor: isBusy ? '#94a3b8' : !selectedFile || !selectedCameraId ? '#cbd5e1' : '#003366',
                color: '#ffffff',
                border: 'none',
                fontSize: '13.5px',
                fontWeight: 800,
                letterSpacing: '0.04em',
                cursor: isBusy || !selectedFile || !selectedCameraId ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: isBusy || !selectedFile ? 'none' : '0 2px 6px rgba(0, 51, 102, 0.3)',
                transition: 'all 0.15s ease'
              }}
            >
              <span
                className={`material-symbols-outlined ${isBusy ? 'animate-spin' : ''}`}
                style={{ fontSize: '18px' }}
              >
                {isBusy ? 'sync' : 'play_arrow'}
              </span>
              <span>{isBusy ? 'EXECUTING AI PIPELINE...' : 'START AI PROCESSING'}</span>
            </button>
          </div>

          {/* Card 3: Execution State & Stepper */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="material-symbols-outlined" style={{ color: '#003366', fontSize: '20px' }}>account_tree</span>
                <h2 style={{ fontSize: '14px', fontWeight: 800, color: '#001e40', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Pipeline Stages
                </h2>
              </div>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#003366' }}>
                {jobState.progress_percent}%
              </span>
            </div>

            {/* Linear Progress Bar */}
            <div style={{ width: '100%', height: '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden', marginBottom: '12px' }}>
              <div style={{
                width: `${jobState.progress_percent}%`,
                height: '100%',
                backgroundColor: jobState.status === 'FAILED' ? '#dc2626' : jobState.status === 'COMPLETED' ? '#16a34a' : '#003366',
                transition: 'width 0.4s ease'
              }} />
            </div>

            {/* Active Stage Label Notice */}
            {jobState.stage_label && (
              <div style={{
                fontSize: '11.5px',
                fontWeight: 600,
                color: jobState.status === 'FAILED' ? '#dc2626' : '#003366',
                backgroundColor: jobState.status === 'FAILED' ? 'rgba(220, 38, 38, 0.08)' : 'rgba(0, 51, 102, 0.05)',
                padding: '6px 10px',
                borderRadius: '5px',
                marginBottom: '12px'
              }}>
                {jobState.stage_label}
              </div>
            )}

            {/* Stepper Checklist */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {STAGES.map((st, idx) => {
                const stageOrder = ['VIDEO_LOADING', 'MODEL_INITIALIZATION', 'AI_PERCEPTION', 'BYTETRACK_TRACKING', 'PADDLE_OCR', 'REID_EMBEDDING', 'DATABASE_INGESTION'];
                const currentStageIdx = stageOrder.indexOf(jobState.stage);
                const thisIdx = idx;

                let isCompleted = false;
                let isCurrent = false;

                if (jobState.status === 'COMPLETED') {
                  isCompleted = true;
                } else if (jobState.status === 'PROCESSING') {
                  if (thisIdx < currentStageIdx) isCompleted = true;
                  else if (thisIdx === currentStageIdx) isCurrent = true;
                }

                return (
                  <div
                    key={st.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '5px 8px',
                      borderRadius: '4px',
                      backgroundColor: isCurrent ? 'rgba(0, 51, 102, 0.06)' : 'transparent',
                      color: isCompleted ? '#16a34a' : isCurrent ? '#003366' : '#94a3b8',
                      fontSize: '11.5px',
                      fontWeight: isCurrent || isCompleted ? 700 : 500
                    }}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>
                      {isCompleted ? 'check_circle' : isCurrent ? 'sync' : 'radio_button_unchecked'}
                    </span>
                    <span>{st.label}</span>
                  </div>
                );
              })}
            </div>
          </div>

        </div>

        {/* ========================================================= */}
        {/* RIGHT COLUMN: PROCESSED VIDEO & REAL TELEMETRY RESULTS    */}
        {/* ========================================================= */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

          {/* Card A: Annotated Processed Video Player */}
          <div style={{
            backgroundColor: '#ffffff',
            borderRadius: '8px',
            border: '1px solid #d0dbe7',
            boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
            padding: '1.25rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="material-symbols-outlined" style={{ color: '#003366', fontSize: '20px' }}>preview</span>
                <h2 style={{ fontSize: '14px', fontWeight: 800, color: '#001e40', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  AI Annotated Video Output
                </h2>
              </div>

              {jobState.status === 'COMPLETED' && (
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#16a34a', backgroundColor: 'rgba(22, 163, 74, 0.1)', padding: '2px 8px', borderRadius: '4px' }}>
                  H.264 Web Stream Ready
                </span>
              )}
            </div>

            {/* Video Canvas / Player Area */}
            <div style={{
              width: '100%',
              minHeight: '380px',
              backgroundColor: '#0a192f',
              borderRadius: '6px',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative'
            }}>
              {jobState.status === 'COMPLETED' && jobState.video_url ? (
                <video
                  key={jobState.video_url}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  style={{ width: '100%', maxHeight: '540px', objectFit: 'contain' }}
                  src={jobState.video_url}
                >
                  Your browser does not support HTML5 video streaming.
                </video>
              ) : isBusy ? (
                <div style={{ textAlign: 'center', color: '#94a3b8', padding: '2rem' }}>
                  <div style={{
                    width: '48px',
                    height: '48px',
                    border: '3px solid rgba(56, 189, 248, 0.2)',
                    borderTop: '3px solid #38bdf8',
                    borderRadius: '50%',
                    margin: '0 auto 1rem auto',
                    animation: 'spin 1s linear infinite'
                  }} />
                  <div style={{ fontSize: '14px', fontWeight: 700, color: '#ffffff' }}>
                    Executing AI Neural Models...
                  </div>
                  <div style={{ fontSize: '11.5px', color: '#94a3b8', marginTop: '4px' }}>
                    {jobState.stage_label || 'Detecting vehicles, tracking bounding boxes, and reading plates'}
                  </div>
                </div>
              ) : jobState.status === 'FAILED' ? (
                <div style={{ textAlign: 'center', color: '#f87171', padding: '2rem' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '48px', color: '#ef4444', marginBottom: '8px' }}>error</span>
                  <div style={{ fontSize: '14px', fontWeight: 700 }}>Processing Failed</div>
                  <div style={{ fontSize: '11.5px', color: '#cbd5e1', marginTop: '4px', maxWidth: '420px', margin: '4px auto 0 auto' }}>
                    {jobState.error || 'An unexpected error occurred during execution.'}
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem' }}>
                  <span className="material-symbols-outlined" style={{ fontSize: '48px', color: '#475569', marginBottom: '8px' }}>videocam</span>
                  <div style={{ fontSize: '13.5px', fontWeight: 700, color: '#cbd5e1' }}>
                    Ready for Video Input
                  </div>
                  <div style={{ fontSize: '11.5px', color: '#94a3b8', marginTop: '4px' }}>
                    Select a camera node and upload traffic footage to view annotated output.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Card B: Real AI Telemetry Metrics */}
          {jobState.metrics && (
            <div style={{
              backgroundColor: '#ffffff',
              borderRadius: '8px',
              border: '1px solid #d0dbe7',
              boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
              padding: '1.25rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span className="material-symbols-outlined" style={{ color: '#003366', fontSize: '20px' }}>analytics</span>
                  <h2 style={{ fontSize: '14px', fontWeight: 800, color: '#001e40', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Real AI Perception Metrics
                  </h2>
                </div>

                <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 600 }}>
                  Device: <strong style={{ color: '#003366' }}>{jobState.metrics.device?.toUpperCase() || 'GPU'}</strong> • Time: <strong style={{ color: '#003366' }}>{jobState.metrics.processing_time_sec}s</strong> ({jobState.metrics.pipeline_fps} FPS)
                </div>
              </div>

              {/* 6 Grid Metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '10px', marginBottom: '16px' }}>
                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>Vehicles Detected</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#003366', marginTop: '2px' }}>
                    {jobState.metrics.vehicles_detected ?? 'N/A'}
                  </div>
                </div>

                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>ByteTrack Tracks</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#00677d', marginTop: '2px' }}>
                    {jobState.metrics.vehicle_tracks ?? 'N/A'}
                  </div>
                </div>

                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>Plate Detections</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#0284c7', marginTop: '2px' }}>
                    {jobState.metrics.plate_detections ?? 'N/A'}
                  </div>
                </div>

                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>OCR Reads</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#16a34a', marginTop: '2px' }}>
                    {jobState.metrics.successful_ocr_reads ?? 'N/A'}
                  </div>
                </div>

                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>Re-ID Embeddings</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#7c3aed', marginTop: '2px' }}>
                    {jobState.metrics.reid_embeddings ?? 'N/A'}
                  </div>
                </div>

                <div style={{ padding: '10px', borderRadius: '6px', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase' }}>DB Observations</div>
                  <div style={{ fontSize: '20px', fontWeight: 900, color: '#ea580c', marginTop: '2px' }}>
                    {jobState.metrics.db_ingested_observations ?? 'N/A'}
                  </div>
                </div>
              </div>

              {/* Detections Records List */}
              {Array.isArray(jobState.metrics.detections) && jobState.metrics.detections.length > 0 && (
                <div>
                  <div style={{ fontSize: '12px', fontWeight: 800, color: '#1e293b', marginBottom: '8px' }}>
                    Confirmed Track Observations ({jobState.metrics.detections.length})
                  </div>
                  <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11.5px', textAlign: 'left' }}>
                      <thead style={{ backgroundColor: '#f8fafc', position: 'sticky', top: 0, zIndex: 10 }}>
                        <tr style={{ borderBottom: '1px solid #cbd5e1' }}>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Track ID</th>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Vehicle Class</th>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Plate OCR</th>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Status</th>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Frames</th>
                          <th style={{ padding: '6px 10px', color: '#475569' }}>Re-ID Samples</th>
                        </tr>
                      </thead>
                      <tbody>
                        {jobState.metrics.detections.map((d) => (
                          <tr key={d.track_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 10px', fontWeight: 700, color: '#003366' }}>
                              #{d.track_id}
                            </td>
                            <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>
                              {d.vehicle_class}
                            </td>
                            <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontWeight: 700 }}>
                              {d.plate_text || <span style={{ color: '#94a3b8' }}>None</span>}
                            </td>
                            <td style={{ padding: '6px 10px' }}>
                              <span style={{
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontSize: '10px',
                                fontWeight: 700,
                                backgroundColor: d.plate_status === 'stable' ? 'rgba(22, 163, 74, 0.1)' : 'rgba(148, 163, 184, 0.15)',
                                color: d.plate_status === 'stable' ? '#16a34a' : '#64748b'
                              }}>
                                {d.plate_status}
                              </span>
                            </td>
                            <td style={{ padding: '6px 10px', color: '#64748b' }}>
                              {d.frame_count}
                            </td>
                            <td style={{ padding: '6px 10px', color: '#7c3aed', fontWeight: 600 }}>
                              {d.reid_count}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Cross-Navigation Actions */}
                  <div style={{ display: 'flex', gap: '10px', marginTop: '14px', flexWrap: 'wrap' }}>
                    {setRoute && (
                      <>
                        <button
                          onClick={() => setRoute('/tracking')}
                          style={{
                            padding: '6px 12px',
                            borderRadius: '6px',
                            backgroundColor: '#f1f5f9',
                            border: '1px solid #cbd5e1',
                            fontSize: '11.5px',
                            fontWeight: 700,
                            color: '#003366',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>radar</span>
                          View in Vehicle Tracking
                        </button>

                        <button
                          onClick={() => setRoute('/anpr')}
                          style={{
                            padding: '6px 12px',
                            borderRadius: '6px',
                            backgroundColor: '#f1f5f9',
                            border: '1px solid #cbd5e1',
                            fontSize: '11.5px',
                            fontWeight: 700,
                            color: '#003366',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}
                        >
                          <span className="material-symbols-outlined" style={{ fontSize: '15px' }}>document_scanner</span>
                          View in ANPR Captures
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

        </div>

      </div>
    </div>
  );
}
