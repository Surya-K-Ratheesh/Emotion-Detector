"use client";

import dynamic from "next/dynamic";
import styles from "./page.module.css";

// Dynamically import EmotionDetector with SSR disabled
const EmotionDetector = dynamic(() => import("@/components/EmotionDetector"), {
  ssr: false,
  loading: () => (
    <div style={{ 
      display: "flex", 
      justifyContent: "center", 
      alignItems: "center", 
      minHeight: "400px",
      color: "#a4b0be",
      fontFamily: "system-ui, sans-serif"
    }}>
      Loading webcam interface...
    </div>
  )
});

export default function Home() {
  const techStack = [
    "Python",
    "Keras",
    "OpenCV",
    "TensorFlow.js",
    "Next.js",
    "Vercel"
  ];

  return (
    <div className={styles.page}>
      <main className={styles.main}>
        {/* Header Branding */}
        <section className={styles.headerSection}>
          <h1 className={styles.titleGlow}>
            Real-Time Facial Emotion Recognition
          </h1>
          <p className={styles.description}>
            This application uses a custom-trained deep CNN model running entirely on the client side. 
            A face detector extracts bounding boxes, which are normalized into 48x48 grayscale inputs, 
            and classified into 7 core human emotions in real time.
          </p>
          <div className={styles.techStack}>
            {techStack.map((tech) => (
              <span key={tech} className={styles.techBadge}>
                {tech}
              </span>
            ))}
          </div>
        </section>

        {/* Emotion Recognition Web Interface */}
        <EmotionDetector />

        {/* Footer Technical details */}
        <footer className={styles.footer}>
          <p>
            Powered by WebGL Acceleration inside TensorFlow.js. All processing is local and secure.
          </p>
        </footer>
      </main>
    </div>
  );
}
