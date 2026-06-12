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


export default function EmotionDetector() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const emotionModelRef = useRef<tf.LayersModel | null>(null);
  const animationFrameId = useRef<number | null>(null);

  // Status and loaded states
  const [status, setStatus] = useState<string>("Initializing...");
  const [isLoaded, setIsLoaded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isActive] = useState<boolean>(true);

  // Statistics & Real-time prediction states
  // HUD Tracking box state
  const [trackingBox, setTrackingBox] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
    color: string;
    emotionLabel: string;
    confidence: number;
    distribution: Record<string, number>;
    rawX: number;
    rawY: number;
  } | null>(null);

  // Video resolution
  const videoConstraints = {
    width: 1280,
    height: 720,
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
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models/faceapi");

        setStatus("Loading Emotion Classifier...");
        // Load custom TFJS model from static /models/emotion directory
        const loadedModel = await tf.loadLayersModel("/models/emotion/model.json");
        emotionModelRef.current = loadedModel;

        if (active) {
          setStatus("Models loaded successfully. Camera initializing...");
          setIsLoaded(true);
        }
      } catch (err: unknown) {
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

      // Draw a bright red border to verify if the canvas layer is visible on screen
      ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
      ctx.lineWidth = 8;
      ctx.strokeRect(0, 0, canvas.width, canvas.height);

      try {
        // 1. Detect single face bounding box
        const detection = await faceapi.detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.25 })
        );

        if (detection) {
          const { x, y, width: w, height: h } = detection.box;

          // Align crop to be a bit more square and padded (standard for training images)
          const padX = w * 0.1;
          const padY = h * 0.1;
          const cropX = Math.max(0, x - padX);
          const cropY = Math.max(0, y - padY);
          const cropW = Math.min(videoWidth - cropX, w + 2 * padX);
          const cropH = Math.min(videoHeight - cropY, h + 2 * padY);

          // Tight bounding box for visual overlay (hairline to chin)
          const tightX = x + w * 0.15;
          const tightY = y + h * 0.1;
          const tightW = w * 0.7;
          const tightH = h * 0.85;

          // Responsive Percentages (mirrored X to match CSS transform scaleX(-1))
          const leftPercent = (tightX / videoWidth) * 100;
          const topPercent = (tightY / videoHeight) * 100;
          const widthPercent = (tightW / videoWidth) * 100;
          const heightPercent = (tightH / videoHeight) * 100;
          const mirroredLeftPercent = 100 - leftPercent - widthPercent;

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

            // 4. Update HUD tracking state instead of drawing on canvas
            setTrackingBox({
              x: mirroredLeftPercent,
              y: topPercent,
              w: widthPercent,
              h: heightPercent,
              color: EMOTION_COLORS[EMOTIONS[maxIdx]],
              emotionLabel: EMOTIONS[maxIdx],
              confidence: maxVal,
              distribution: distribution,
              rawX: Math.round(x),
              rawY: Math.round(y)
            });
            
          }
        } else {
          // Reset predictions to zero when no face is present
          setTrackingBox(null);
        }
      } catch (err) {
        console.error("Frame processing error:", err);
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

        {/* HUD Diagnostic Overlay */}
        {isLoaded && (
          <div style={{
            position: "absolute",
            top: "12px",
            left: "12px",
            background: "rgba(10, 10, 15, 0.85)",
            backdropFilter: "blur(4px)",
            color: "#a4b0be",
            padding: "8px 12px",
            borderRadius: "8px",
            fontSize: "11px",
            fontFamily: "monospace",
            zIndex: 100,
            border: "1px solid rgba(255, 255, 255, 0.15)",
            pointerEvents: "none"
          }}>
            Detector Sensitivity: 0.25<br />
            System Status: {isActive ? "Active" : "Paused"} | {trackingBox ? "Face DETECTED" : "Searching..."}
          </div>
        )}

        {/* Futuristic HTML Tracking Box and HUD */}
        {trackingBox && (
          <div
            style={{
              position: "absolute",
              left: `${trackingBox.x}%`,
              top: `${trackingBox.y}%`,
              width: `${trackingBox.w}%`,
              height: `${trackingBox.h}%`,
              border: `2px solid ${trackingBox.color}`,
              boxShadow: `0 0 15px ${trackingBox.color}, inset 0 0 15px ${trackingBox.color}`,
              pointerEvents: "none",
              zIndex: 50,
            }}
          >
            {/* Tracking Corners */}
            <div className={`${styles.trackingCorner} ${styles.trackingCornerTopLeft}`} style={{ borderColor: trackingBox.color }}></div>
            <div className={`${styles.trackingCorner} ${styles.trackingCornerTopRight}`} style={{ borderColor: trackingBox.color }}></div>
            <div className={`${styles.trackingCorner} ${styles.trackingCornerBottomLeft}`} style={{ borderColor: trackingBox.color }}></div>
            <div className={`${styles.trackingCorner} ${styles.trackingCornerBottomRight}`} style={{ borderColor: trackingBox.color }}></div>

            {/* Crosshairs */}
            <div className={`${styles.hudCrosshair} ${styles.pulseAnim}`} style={{ top: "0%", left: "50%", borderColor: trackingBox.color }}></div>
            <div className={`${styles.hudCrosshair} ${styles.pulseAnim}`} style={{ top: "100%", left: "50%", borderColor: trackingBox.color }}></div>
            <div className={`${styles.hudCrosshair} ${styles.pulseAnim}`} style={{ top: "50%", left: "0%", borderColor: trackingBox.color }}></div>
            <div className={`${styles.hudCrosshair} ${styles.pulseAnim}`} style={{ top: "50%", left: "100%", borderColor: trackingBox.color }}></div>
            
            {/* Multi-component Unified Data Box Anchored Above */}
            <div className={styles.multiComponentDataBox} style={{ borderColor: trackingBox.color }}>
              
              {/* Connecting line to the box */}
              <div className={styles.multiComponentConnectingLine} style={{ backgroundColor: trackingBox.color }}></div>

              {/* ID Badge */}
              <div className={styles.hudIdBadge} style={{ background: trackingBox.color, color: "#000" }}>
                ID: 001
              </div>

              {/* Top HUD Panel */}
              <div className={styles.hudPanel} style={{ borderColor: trackingBox.color, boxShadow: `0 10px 20px rgba(0,0,0,0.3)` }}>
                <div className={styles.hudTitle} style={{ color: trackingBox.color }}>
                  {trackingBox.emotionLabel}
                </div>
                
                <div className={styles.hudBars}>
                  {Object.entries(trackingBox.distribution)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 3)
                    .map(([emotion, prob]) => {
                      const isTop = emotion === trackingBox.emotionLabel;
                      return (
                        <div key={emotion} className={styles.hudBarRow}>
                          <div className={styles.hudBarBg}>
                            <div 
                              className={styles.hudBarFill} 
                              style={{ 
                                width: `${prob * 100}%`,
                                background: isTop ? trackingBox.color : EMOTION_COLORS[emotion] || "#a4b0be",
                                boxShadow: `0 0 5px ${isTop ? trackingBox.color : "transparent"}`
                              }}
                            ></div>
                          </div>
                          <div className={styles.hudDataValue}>
                            {Math.round(prob * 100)}%
                          </div>
                          <div className={styles.hudLabel} style={{ color: isTop ? trackingBox.color : "#a4b0be" }}>
                            {emotion.toUpperCase()}
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Coordinates Block */}
              <div className={styles.hudCoordinatesPanel} style={{ borderColor: trackingBox.color, color: trackingBox.color }}>
                x: {trackingBox.rawX}<br />
                y: {trackingBox.rawY}<br />
                Confidence: {Math.round(trackingBox.confidence * 100)}%
              </div>

            </div>
          </div>
        )}
        
        <Webcam
          ref={webcamRef}
          audio={false}
          muted={true}
          videoConstraints={videoConstraints}
          className={styles.webcam}
        />
        
        <canvas ref={canvasRef} className={styles.overlayCanvas} />
      </div>
    </div>
  );
}
