"use client";

import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as faceapi from "@vladmandic/face-api";
import styles from "./EmotionDetector.module.css";

const EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"];

// Beautiful neon colors for each emotion
const EMOTION_COLORS: Record<string, string> = {
  Angry: "#ff4757",
  Disgust: "#2ed573",
  Fear: "#a4b0be",
  Happy: "#ffa502",
  Sad: "#1e90ff",
  Surprise: "#eccc68",
  Neutral: "#70a1ff"
};

// Emojis mapping
const EMOTION_EMOJIS: Record<string, string> = {
  Angry: "😠",
  Disgust: "🤢",
  Fear: "😨",
  Happy: "😊",
  Sad: "😢",
  Surprise: "😲",
  Neutral: "😐"
};

export default function EmotionDetector() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const emotionModelRef = useRef<tf.LayersModel | null>(null);
  const animationFrameId = useRef<number | null>(null);

  // Status and loaded states
  const [status, setStatus] = useState<string>("Initializing...");
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isActive, setIsActive] = useState<boolean>(true);

  // Statistics & Real-time prediction states
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [dominantEmotion, setDominantEmotion] = useState<{ label: string; confidence: number }>({
    label: "Neutral",
    confidence: 0
  });
  const [emotionDistribution, setEmotionDistribution] = useState<Record<string, number>>(
    EMOTIONS.reduce((acc, curr) => ({ ...acc, [curr]: 0 }), {})
  );

  // Video resolution
  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  useEffect(() => {
    let active = true;

    async function loadModels() {
      try {
        setStatus("Initializing TensorFlow.js...");
        await tf.ready();

        setStatus("Loading Face Detection model...");
        // Load tinyFaceDetector weights from the static /models/faceapi directory
        const faceApiOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models/faceapi");

        setStatus("Loading Emotion Classifier...");
        // Load custom TFJS model from static /models/emotion directory
        const loadedModel = await tf.loadLayersModel("/models/emotion/model.json");
        emotionModelRef.current = loadedModel;

        if (active) {
          setStatus("Models loaded successfully. Camera initializing...");
          setIsLoaded(true);
        }
      } catch (err: any) {
        console.error("Error loading models:", err);
        if (active) {
          setError(
            "Failed to load AI models. Ensure you ran the model setup script and served public assets."
          );
        }
      }
    }

    loadModels();

    return () => {
      active = false;
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      if (emotionModelRef.current) {
        emotionModelRef.current.dispose();
      }
    };
  }, []);

  // Frame processing loop
  useEffect(() => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsInterval = performance.now();

    async function processFrame() {
      if (!isActive || !isLoaded || !emotionModelRef.current) {
        animationFrameId.current = requestAnimationFrame(processFrame);
        return;
      }

      const webcam = webcamRef.current;
      const canvas = canvasRef.current;

      if (!webcam || !webcam.video || webcam.video.readyState !== 4 || !canvas) {
        animationFrameId.current = requestAnimationFrame(processFrame);
        return;
      }

      const video = webcam.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Match canvas dimensions to the video
      if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        animationFrameId.current = requestAnimationFrame(processFrame);
        return;
      }

      // Clear the overlay canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        // 1. Detect single face bounding box
        const detection = await faceapi.detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
        );

        if (detection) {
          const t0 = performance.now();
          const { x, y, width: w, height: h } = detection.box;

          // Align crop to be a bit more square and padded (standard for training images)
          const padX = w * 0.1;
          const padY = h * 0.1;
          const cropX = Math.max(0, x - padX);
          const cropY = Math.max(0, y - padY);
          const cropW = Math.min(videoWidth - cropX, w + 2 * padX);
          const cropH = Math.min(videoHeight - cropY, h + 2 * padY);

          // 2. Preprocess cropped face to 48x48 Grayscale image
          const cropCanvas = document.createElement("canvas");
          cropCanvas.width = 48;
          cropCanvas.height = 48;
          const cropCtx = cropCanvas.getContext("2d");
          
          if (cropCtx) {
            // Draw cropped video segment onto the 48x48 canvas
            cropCtx.drawImage(video, cropX, cropY, cropW, cropH, 0, 0, 48, 48);

            // Convert to grayscale & run prediction in tf.tidy to prevent memory leaks
            const predictions = tf.tidy(() => {
              // Convert 48x48 canvas to tensor (shape: [48, 48, 3])
              const imgTensor = tf.browser.fromPixels(cropCanvas);
              
              // Convert to grayscale: compute mean across channels (shape: [48, 48, 1])
              const grayTensor = imgTensor.mean(2).expandDims(-1);
              
              // Normalize pixels to [0.0, 1.0] and add batch dimension (shape: [1, 48, 48, 1])
              const preprocessed = grayTensor.toFloat().div(tf.scalar(255.0)).expandDims(0);
              
              // Run prediction
              return emotionModelRef.current!.predict(preprocessed) as tf.Tensor;
            });

            // Retrieve probability values asynchronously
            const probs = await predictions.data();
            predictions.dispose();

            const t1 = performance.now();
            setInferenceTime(Math.round(t1 - t0));

            // 3. Compute dominant emotion and distribution state
            const distribution: Record<string, number> = {};
            let maxIdx = 0;
            let maxVal = -1;

            EMOTIONS.forEach((emotion, idx) => {
              const val = probs[idx] || 0;
              distribution[emotion] = val;
              if (val > maxVal) {
                maxVal = val;
                maxIdx = idx;
              }
            });

            setEmotionDistribution(distribution);
            setDominantEmotion({
              label: EMOTIONS[maxIdx],
              confidence: maxVal
            });

            // 4. Render Bounding Box and Overlay on Canvas
            const boxColor = EMOTION_COLORS[EMOTIONS[maxIdx]];
            
            // Draw Box with Neon Glow
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 4;
            ctx.shadowColor = boxColor;
            ctx.shadowBlur = 15;
            
            // Draw rounded rectangle
            const radius = 12;
            ctx.beginPath();
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + w - radius, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
            ctx.lineTo(x + w, y + h - radius);
            ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
            ctx.lineTo(x + radius, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
            ctx.lineTo(x, y + radius);
            ctx.quadraticCurveTo(x, y, x + radius, y);
            ctx.closePath();
            ctx.stroke();

            // Text Label Box
            ctx.shadowBlur = 0; // Disable shadow for text readability
            ctx.fillStyle = "rgba(10, 10, 10, 0.75)";
            const text = `${EMOTION_EMOJIS[EMOTIONS[maxIdx]]} ${EMOTIONS[maxIdx]} (${Math.round(maxVal * 100)}%)`;
            ctx.font = "bold 16px sans-serif";
            const textWidth = ctx.measureText(text).width;
            
            ctx.fillRect(x - 2, y - 35, textWidth + 20, 30);
            
            // Text Color Accent
            ctx.fillStyle = boxColor;
            ctx.fillText(text, x + 8, y - 14);
          }
        } else {
          // Fade predictions when no face is present
          setDominantEmotion(prev => ({ ...prev, confidence: prev.confidence * 0.9 }));
        }
      } catch (err) {
        console.error("Frame processing error:", err);
      }

      // Compute Frame Rate (FPS)
      frameCount++;
      const now = performance.now();
      if (now - fpsInterval >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsInterval)));
        frameCount = 0;
        fpsInterval = now;
      }

      animationFrameId.current = requestAnimationFrame(processFrame);
    }

    if (isLoaded) {
      animationFrameId.current = requestAnimationFrame(processFrame);
    }

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [isLoaded, isActive]);

  const toggleTracking = () => {
    setIsActive(!isActive);
    if (!isActive) {
      setStatus("Tracking active");
    } else {
      setStatus("Tracking paused");
      // Clear canvas overlay when tracking paused
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <div className={styles.errorIcon}>⚠️</div>
        <h3>Initialization Failed</h3>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Webcam Frame Panel */}
      <div className={styles.videoWrapper}>
        {!isLoaded && (
          <div className={styles.loaderOverlay}>
            <div className={styles.spinner}></div>
            <p className={styles.loadingText}>{status}</p>
          </div>
        )}
        
        {/* Animated Scanning Line overlay */}
        {isLoaded && isActive && <div className={styles.scannerLine} />}

        <Webcam
          ref={webcamRef}
          audio={false}
          muted={true}
          videoConstraints={videoConstraints}
          className={styles.webcam}
        />
        
        <canvas ref={canvasRef} className={styles.overlayCanvas} />
      </div>

      {/* Dashboard Analytics Panel */}
      <div className={styles.dashboard}>
        <div className={styles.header}>
          <h2>Live Analytics</h2>
          <div className={styles.stats}>
            <span className={`${styles.badge} ${isActive ? styles.badgeActive : styles.badgeInactive}`}>
              {isActive ? "LIVE SCANNING" : "PAUSED"}
            </span>
            {isLoaded && (
              <>
                <span className={styles.badge}>Inference: {inferenceTime}ms</span>
                <span className={styles.badge}>FPS: {fps}</span>
              </>
            )}
          </div>
        </div>

        {/* Dominant Emotion Output */}
        <div 
          className={styles.dominantPanel} 
          style={{ borderColor: dominantEmotion.confidence > 0.1 ? EMOTION_COLORS[dominantEmotion.label] : "#222" }}
        >
          <div className={styles.dominantEmoji}>
            {dominantEmotion.confidence > 0.1 ? EMOTION_EMOJIS[dominantEmotion.label] : "🔍"}
          </div>
          <div className={styles.dominantMeta}>
            <span className={styles.label}>DOMINANT EXPRESSION</span>
            <span className={styles.value}>
              {dominantEmotion.confidence > 0.1 ? dominantEmotion.label : "Detecting..."}
            </span>
            <span className={styles.confidence}>
              {dominantEmotion.confidence > 0.1 ? `${Math.round(dominantEmotion.confidence * 100)}% Confidence` : "Scan your face"}
            </span>
          </div>
        </div>

        {/* Emotion Distribution Bars */}
        <div className={styles.distributionContainer}>
          <h3>Probability Distribution</h3>
          <div className={styles.barsList}>
            {EMOTIONS.map((emotion) => {
              const confidence = emotionDistribution[emotion] || 0;
              const color = EMOTION_COLORS[emotion];
              return (
                <div key={emotion} className={styles.barItem}>
                  <div className={styles.barHeader}>
                    <span>{EMOTION_EMOJIS[emotion]} {emotion}</span>
                    <span>{Math.round(confidence * 100)}%</span>
                  </div>
                  <div className={styles.barBackground}>
                    <div 
                      className={styles.barFill} 
                      style={{ 
                        width: `${confidence * 100}%`,
                        backgroundColor: color,
                        boxShadow: `0 0 8px ${color}`
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Controls */}
        <div className={styles.controls}>
          <button 
            onClick={toggleTracking} 
            disabled={!isLoaded}
            className={`${styles.btn} ${isActive ? styles.btnPause : styles.btnPlay}`}
          >
            {isActive ? "Pause Analysis" : "Resume Analysis"}
          </button>
        </div>
      </div>
    </div>
  );
}
