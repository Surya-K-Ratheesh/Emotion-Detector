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
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        {/* Header Branding */}
        <section className={styles.headerSection}>
          <h1 className={styles.titleGlow}>
            Project Manas
          </h1>
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
